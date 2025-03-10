import numpy as np
cimport numpy as np
from numpy.linalg import inv
from polymer.common import L2FLAGS
from libc.math cimport nan, exp, log, abs, sqrt, isnan
from cpython.exc cimport PyErr_CheckSignals
import pandas as pd
from pathlib import Path

from polymer.neldermead cimport NelderMeadMinimizer, dot
from polymer.water cimport WaterModel
from polymer.glint import glitter

'''
main polymer iterative optimization module
'''

cdef enum METRICS:
    W_dR2_norm = 1
    W_absdR = 2
    W_absdR_norm = 3
    W_absdR_Rprime = 4
    W_absdR2_Rprime2 = 5
    W_dR2_Rprime_noglint2 = 6
    polymer_3_5 = 7
    W_dR2_Rprime_noglint2_norm = 8

metrics_names = {
        'W_dR2_norm': W_dR2_norm,
        'W_absdR': W_absdR,
        'W_absdR_norm': W_absdR_norm,
        'W_absdR_Rprime': W_absdR_Rprime,
        'W_absdR2_Rprime2': W_absdR2_Rprime2,
        'W_dR2_Rprime_noglint2': W_dR2_Rprime_noglint2,
        'W_dR2_Rprime_noglint2_norm': W_dR2_Rprime_noglint2_norm,
        'polymer_3_5': polymer_3_5,
        }

cdef class F(NelderMeadMinimizer):
    '''
    Defines the cost function minimized by Polymer
    Inherits from NelderMeadMinimizer which provides method minimize
    '''

    cdef float[:] Rprime
    cdef float[:] Rprime_noglint
    cdef float[:] Tmol
    cdef float[:] wav
    cdef WaterModel w

    # [Ratm] = [A] . [C]
    # where A is the matrix of the polynomial exponents for each wavelength (nlam x ncoef)
    # [C] = [pA] . [Ratm]    where [pA] is the pseudoinverse of matrix [A]  (ncoef x nlam)
    cdef float[:,:] A
    cdef float[:,:] pA
    cdef int Ncoef
    cdef float thres_chi2
    cdef float constraint_amplitude, sigma1, sigma2

    cdef float[:] C  # ci coefficients (ncoef)
    cdef float[:] Rwmod
    cdef float[:] Ratm

    # bands
    cdef int N_bands_corr
    cdef int[:] i_corr_read  # index or the 'corr' bands within the 'read' bands
    cdef int N_bands_oc
    cdef int[:] i_oc_read  # index or the 'oc' bands within the 'read' bands
    cdef int N_bands_read
    cdef float[:] weights_oc

    cdef METRICS metrics

    def __init__(self, Ncoef, watermodel, params, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)

        self.w = watermodel
        self.C = np.zeros(Ncoef, dtype='float32')
        self.Ratm = np.zeros(len(params.bands_read()), dtype='float32') + np.nan
        self.Ncoef = Ncoef

        self.thres_chi2 = params.thres_chi2
        self.constraint_amplitude, self.sigma2, self.sigma1 = params.constraint_logfb

        self.N_bands_corr = len(params.bands_corr)
        self.i_corr_read = np.searchsorted(
                params.bands_read(),
                params.bands_corr).astype('int32')
        self.N_bands_oc = len(params.bands_oc)
        self.i_oc_read = np.searchsorted(
                params.bands_read(),
                params.bands_oc).astype('int32')
        self.N_bands_read = len(params.bands_read())
        if params.weights_oc is None:
            self.weights_oc = np.ones(len(params.bands_oc), dtype='float32')
        else:
            assert len(params.weights_oc) == len(params.bands_oc)
            self.weights_oc = np.array(params.weights_oc, dtype='float32')

        try:
            self.metrics = metrics_names[params.metrics]
        except KeyError:
            raise Exception('Invalid metrics "{}"'.format(params.metrics))


    cdef int init_pixel(self, float[:] Rprime, float[:] Rprime_noglint,
                   float[:,:] A, float[:,:] pA,
                   float[:] Tmol,
                   float[:] wav, float sza, float vza, float raa, float ws) except -1:
        '''
        set the input parameters for the current pixel

        return 1 on error, 0 on success
        '''
        self.Rprime = Rprime
        self.Rprime_noglint = Rprime_noglint
        self.wav = wav  # bands_read
        self.pA = pA
        self.A = A
        self.Tmol = Tmol

        return self.w.init_pixel(wav, sza, vza, raa, ws)


    cdef float eval(self, float[:] x):
        '''
        Evaluate cost function for vector parameters x
        '''
        #
        # calculate the. water reflectance for the current parameters
        # (at bands_read)
        #
        self.Rwmod = self.w.calc_rho(x)

        return self.eval_atm(x)


    cdef float eval_atm(self, float[:] x):
        cdef float C
        cdef float sumsq, sumw, dR, norm
        cdef int icorr, icorr_read
        cdef int ioc, ioc_read, iread
        cdef float sigma

        cdef float[:] Rwmod = self.Rwmod   # TODO: don't use this intermediary variable ?

        #
        # Atmospheric fit
        #
        for ic in range(self.Ncoef):
            C = 0.
            for icorr in range(self.N_bands_corr):
                icorr_read = self.i_corr_read[icorr]
                C += self.pA[ic,icorr] * (self.Rprime[icorr_read]
                                          - self.Tmol[icorr_read]*Rwmod[icorr_read])
            self.C[ic] = C

        #
        # Calculate Ratm
        #
        for iread in range(self.N_bands_read):
            self.Ratm[iread] = 0.
            for ic in range(self.Ncoef):
                self.Ratm[iread] += self.C[ic] * self.A[iread,ic]


        #
        # calculate the residual
        #
        sumsq = 0.
        sumw = 0.
        for ioc in range(self.N_bands_oc):
            ioc_read = self.i_oc_read[ioc]

            dR = self.Rprime[ioc_read]

            # subtract atmospheric signal
            dR -= self.Ratm[ioc_read]

            # divide by transmission
            dR /= self.Tmol[ioc_read]

            dR -= Rwmod[ioc_read]

            norm = Rwmod[ioc_read]
            if norm < self.thres_chi2:
                norm = self.thres_chi2

            if (self.metrics == W_dR2_norm) or (self.metrics == polymer_3_5):
                sumsq += self.weights_oc[ioc]*dR*dR/norm
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR:
                sumsq += self.weights_oc[ioc]*abs(dR)
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR_norm:
                sumsq += self.weights_oc[ioc]*abs(dR)/norm
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR_Rprime:
                sumsq += self.weights_oc[ioc]*abs(dR/self.Rprime[ioc_read])
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR2_Rprime2:
                sumsq += self.weights_oc[ioc]*(dR/self.Rprime[ioc_read])**2
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_dR2_Rprime_noglint2:
                sumsq += self.weights_oc[ioc]*(dR/self.Rprime_noglint[ioc_read])**2
                sumw += self.weights_oc[ioc]

            elif self.metrics ==  W_dR2_Rprime_noglint2_norm:
                sumsq += self.weights_oc[ioc]*(dR/self.Rprime_noglint[ioc_read])**2
                sumw += self.weights_oc[ioc]*(0.001/self.Rprime_noglint[ioc_read])**2

        if self.metrics != polymer_3_5:
            sumsq = sumsq/sumw

        if self.constraint_amplitude != 0:
            # sigma equals sigma1 when chl = 0.01
            # sigma equals sigma2 when chl = 0.1
            sigma = self.sigma1*self.sigma1/self.sigma2*exp(log(self.sigma1/self.sigma2)*x[0])

            sumsq += self.constraint_amplitude * (1. - exp(-x[1]*x[1]/(2*sigma*sigma)))

        return sumsq

def atm_func(block, params, bands):
    '''
    Returns the matrix of coefficients for the atmospheric function
    A [im0, im1, bands, ncoef]

    Ratm = A.C
    Ratm: (shp0, shp1, nlam)
    A   : (shp0, shp1, nlam, ncoef)
    C   : (shp0, shp1, ncoef)
    '''
    # bands for atmospheric fit
    Nlam = len(bands)
    assert Nlam > 0
    shp = block.size
    Ncoef = params.Ncoef   # number of polynomial coefficients
    assert Ncoef > 0

    # correction bands wavelengths
    idx = np.searchsorted(params.bands_read(), bands)
    # transpose: move the wavelength dimension to the end
    lam = block.wavelen[:,:,idx]

    # initialize the matrix for inversion

    taum = 0.00877*((np.array(block.bands)[idx]/1000.)**(-4.05))
    Rgli0 = 0.02
    T0 = np.exp(-taum*((1-0.5*np.exp(-block.Rgli/Rgli0))*block.air_mass)[:,:,None])

    if 'veg' in params.atm_model:
        veg = pd.read_csv(
            Path(params.dir_common)/'vegetation.grass.avena.fatua.vswir.vh352.ucsb.asd.spectrum.txt',
            skiprows=21,
            sep=None,
            names=['wav_um', 'r_percent'],
            index_col=0,
            engine='python').to_xarray()
        veg_interpolated = (veg.r_percent/100.).interp(wav_um=lam.ravel()/1000.).values.reshape(lam.shape)

    if params.atm_model == 'T0,-1,-4':
        assert Ncoef == 3
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = (lam/1000.)**-4.
    elif params.atm_model == 'T0,-1,Rmol':
        assert Ncoef == 3
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = block.Rmol[:,:,idx]
    elif params.atm_model == 'T0,-1':
        assert Ncoef == 2
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
    elif params.atm_model == 'T0,-2':
        assert Ncoef == 2
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-2.
    elif params.atm_model == 'T0,-1,veg':
        assert Ncoef == 3
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = block.Tmol[:,:,idx] * veg_interpolated# * (lam/1000)**-4.
    elif params.atm_model == 'T0,-1,-4,veg':
        assert Ncoef == 4
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = (lam/1000.)**-4.
        A[:,:,:,3] = block.Tmol[:,:,idx] * veg_interpolated# * (lam/1000)**-4.
    else:
        raise Exception('Invalid atmospheric model "{}"'.format(params.atm_model))

    return A

def pseudoinverse(A):
    '''
    Calculate the pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    A* = ((A'.A)^(-1)).A'
    where X' is the transpose of X and X^-1 is the inverse of X

    shapes: A:  [...,i,j]
            A*: [...,j,i]
    '''

    # B = A'.A (with broadcasting)
    B = np.einsum('...ji,...jk->...ik', A, A)

    # check
    # if B.ndim == 4:
        # assert np.allclose(B[0,0,:,:], A[0,0,:,:].transpose().dot(A[0,0,:,:]), equal_nan=True)
        # assert np.allclose(B[-1,0,:,:], A[-1,0,:,:].transpose().dot(A[-1,0,:,:]), equal_nan=True)

    # (B^-1).A' (with broadcasting)
    pA = np.einsum('...ij,...kj->...ik', inv(B), A)

    # check
    # if B.ndim == 4:
        # assert np.allclose(pA[0,0], inv(B[0,0,:,:]).dot(A[0,0,:,:].transpose()), equal_nan=True)
        # assert np.allclose(pA[-1,0], inv(B[-1,0,:,:]).dot(A[-1,0,:,:].transpose()), equal_nan=True)

    return pA


def weighted_pseudoinverse(A, W):
    '''
    Calculate the weighted pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    W is the weight matrix (diagonal)
    A* = ((A'.W.A)^(-1)).A'.W
    '''
    assert W.dtype == 'float32'

    # A'.W.A
    B = np.einsum('...ji,...jk,...kl->...il', A, W, A)

    # (B^-1).A'.W
    pA = np.einsum('...ij,...kj,...kl->...il', inv(B), A, W)

    return pA


cdef int in_bounds(float[:] x, float[:,:] bounds):
    '''
    returns whether vector x (N dimensions) is in bounds (Nx2 dimensions)
    '''
    cdef int r = 1
    cdef int i
    for i in range(x.shape[0]):
        if (x[i] < bounds[i,0]) or (x[i] > bounds[i,1]):
            r = 0
    return r


cdef int raiseflag(unsigned short[:,:] bitmask, int i, int j, int flag):
    if not testflag(bitmask, i, j, flag):
        bitmask[i,j] += flag

cdef int testflag(unsigned short[:,:] bitmask, int i, int j, int flag):
    return bitmask[i,j] & flag != 0

cdef class PolymerMinimizer:

    cdef F f
    cdef int Nparams
    cdef int BITMASK_INVALID
    cdef float NaN
    cdef float[:,:] bounds
    cdef float[:] initial_point_1
    cdef float[:] initial_point_2
    cdef float[:,:] initial_points   # check consistency WRT above
    cdef float[:] initial_step
    cdef float size_end_iter
    cdef int max_iter
    cdef int L2_FLAG_CASE2
    cdef int L2_FLAG_INCONSISTENCY
    cdef int L2_FLAG_THICK_AEROSOL
    cdef int L2_FLAG_OUT_OF_BOUNDS
    cdef int L2_FLAG_EXCEPTION
    cdef int L2_FLAG_ANOMALY_RWMOD_BLUE
    cdef object params
    cdef int normalize
    cdef int force_initialization
    cdef int reinit_rw_neg
    cdef int[:] dbg_pt
    cdef int Rprime_consistency
    cdef int N_bands_oc
    cdef int[:] i_oc_read  # index or the 'oc' bands within the 'read' bands
    cdef int N_bands_read
    cdef int uncertainties
    cdef int Ncoef
    cdef int firstguess_method

    def __init__(self, watermodel, params):

        self.Nparams = len(params.initial_step)
        self.Ncoef = params.Ncoef   # number of atmospheric coefficients
        self.f = F(self.Ncoef, watermodel, params, self.Nparams)
        self.BITMASK_INVALID = params.BITMASK_INVALID
        self.NaN = np.nan

        self.bounds = np.array(params.bounds, dtype='float32')
        self.initial_point_1 = np.array(params.initial_point_1, dtype='float32')
        self.initial_point_2 = np.array(params.initial_point_2, dtype='float32')
        self.initial_points = np.array(params.initial_points, dtype='float32')
        self.initial_step = np.array(params.initial_step, dtype='float32')
        self.size_end_iter = params.size_end_iter
        self.max_iter = params.max_iter
        self.L2_FLAG_CASE2 = L2FLAGS['CASE2']
        self.L2_FLAG_INCONSISTENCY = L2FLAGS['INCONSISTENCY']
        self.L2_FLAG_THICK_AEROSOL = L2FLAGS['THICK_AEROSOL']
        self.L2_FLAG_OUT_OF_BOUNDS = L2FLAGS['OUT_OF_BOUNDS']
        self.L2_FLAG_EXCEPTION = L2FLAGS['EXCEPTION']
        self.L2_FLAG_ANOMALY_RWMOD_BLUE = L2FLAGS['ANOMALY_RWMOD_BLUE']
        self.params = params
        self.uncertainties = params.uncertainties
        self.normalize = params.normalize
        self.force_initialization = params.force_initialization
        self.reinit_rw_neg = params.reinit_rw_neg
        self.dbg_pt = np.array(params.dbg_pt, dtype='int32')
        self.Rprime_consistency = params.Rprime_consistency
        self.firstguess_method = params.firstguess_method

        self.N_bands_oc = len(params.bands_oc)
        self.i_oc_read = np.searchsorted(
                params.bands_read(),
                params.bands_oc).astype('int32')
        self.N_bands_read = len(params.bands_read())


    cdef int loop(self, block,
              float[:,:,:,:] A,
              float[:,:,:,:] pA
              ) except -1:
        '''
        cython method which does the main pixel loop
        (over a block)
        '''

        cdef float[:,:,:] Rprime = block.Rprime
        cdef float[:,:,:] Rprime_noglint = block.Rprime_noglint
        cdef float[:,:] Rnir = block.Rnir
        cdef float[:,:,:] Tmol = block.Tmol
        cdef float[:,:,:] wav = block.wavelen
        cdef float[:] cwav = block.cwavelen
        cdef float[:,:] sza = block.sza
        cdef float[:,:] vza = block.vza
        cdef float[:,:] raa = block.raa
        cdef float[:,:] wind_speed = block.wind_speed.astype('float32')

        cdef unsigned short[:,:] bitmask = block.bitmask
        cdef int Nx = Rprime.shape[0]
        cdef int Ny = Rprime.shape[1]
        cdef int rw_neg

        cdef float[:] x0 = np.zeros(self.Nparams, dtype='float32')
        x0[:] = self.initial_point_1[:]

        # create the output datasets
        block.logchl = np.zeros(block.size, dtype='float32')
        cdef float[:,:] logchl = block.logchl
        block.fa = np.zeros(block.size, dtype='float32')
        cdef float[:,:] fa = block.fa
        block.logfb = np.zeros(block.size, dtype='float32')
        cdef float[:,:] logfb = block.logfb
        block.SPM = np.zeros(block.size, dtype='float32')
        cdef float[:,:] SPM = block.SPM
        block.niter = np.zeros(block.size, dtype='uint32')
        cdef unsigned int[:,:] niter = block.niter
        block.Rw = np.zeros(block.size+(block.nbands,), dtype='float32')
        cdef float[:,:,:] Rw = block.Rw
        block.Ratm = np.zeros(block.size+(block.nbands,), dtype='float32')
        cdef float[:,:,:] Ratm = block.Ratm
        block.Rwmod = np.zeros(block.size+(block.nbands,), dtype='float32') + np.nan
        cdef float[:,:,:] Rwmod = block.Rwmod
        block.eps = np.zeros(block.size, dtype='float32')
        cdef float[:,:] eps = block.eps
        block.Ci = np.zeros(block.size+(self.Ncoef,), dtype='float32')
        cdef float[:,:,:] Ci = block.Ci

        cdef float[:,:] logchl_unc
        cdef float[:,:] logfb_unc
        cdef float[:,:,:] rho_w_unc
        cdef float[:,:,:] Rtoa_var
        cdef float[:,:] rho_w_mod_cov
        cdef float[:,:] d_rw_x_cov
        cdef float[:,:] d_rw_x
        if self.uncertainties:
            block.logchl_unc = np.zeros(block.size, dtype='float32') + np.nan
            logchl_unc = block.logchl_unc
            block.logfb_unc = np.zeros(block.size, dtype='float32') + np.nan
            logfb_unc = block.logfb_unc
            block.rho_w_unc = np.zeros(block.size+(block.nbands,), dtype='float32') + np.nan
            rho_w_unc = block.rho_w_unc
            Rtoa_var = block.Rtoa_var
            d_rw_x = np.zeros((block.nbands, self.Nparams), dtype='float32') + np.nan
            rho_w_mod_cov = np.zeros((block.nbands, block.nbands), dtype='float32') + np.nan
            d_rw_x_cov = np.zeros((block.nbands, self.Nparams), dtype='float32') + np.nan

        cdef int i, j, ib, ioc, iparam
        cdef int flag_reinit = 0
        cdef float Rw_max
        cdef float[:] wav0
        cdef float sza0, vza0, raa0
        cdef float sigmasq
        cdef float delta = 0.05
        cdef float[:,:] Rwmod_fg

        
        if self.initial_points.size:
            Rwmod_fg = np.zeros((self.initial_points.shape[0], block.nbands),
                                dtype='float32') + np.nan
            self.init_first_guess(Rwmod_fg, cwav)

        #
        # pixel loop
        #
        for j in range(Ny):
            for i in range(Nx):

                if (bitmask[i,j] & self.BITMASK_INVALID) != 0:
                    logchl[i,j] = self.NaN
                    fa[i,j] = self.NaN
                    SPM[i,j] = self.NaN
                    logfb[i,j] = self.NaN
                    Rw[i,j,:] = self.NaN
                    Ci[i,j,:] = self.NaN
                    continue

                if self.f.init_pixel(
                        Rprime[i,j,:],
                        Rprime_noglint[i,j,:],
                        A[i,j,:,:], pA[i,j,:,:],
                        Tmol[i,j,:],
                        wav[i,j,:],
                        sza[i,j], vza[i,j], raa[i,j],
                        wind_speed[i,j]):
                    raiseflag(bitmask, i, j, self.L2_FLAG_EXCEPTION)
                    continue

                # first guess
                if self.initial_points.size:
                    self.first_guess(Rwmod_fg, x0, i, j)
                
                self.f.init(x0, self.initial_step)

                # visualization of the cost function
                if self.dbg_pt[0] >= 0:
                    if ((self.dbg_pt[0] == i) and (self.dbg_pt[1] == j)):
                        self.visu_costfunction()
                    else:
                        continue


                # optimization loop
                while self.f.niter < self.max_iter:

                    self.f.iterate()

                    if self.f.size() < self.size_end_iter:
                        break
                    if not in_bounds(self.f.xmin, self.bounds):
                        raiseflag(bitmask, i, j, self.L2_FLAG_CASE2)
                        break

                # case2 optimization if first optimization fails
                if testflag(bitmask, i, j, self.L2_FLAG_CASE2) and (not self.initial_points.size):

                    self.f.init(self.initial_point_2, self.initial_step)

                    while self.f.niter < self.max_iter:

                        self.f.iterate()

                        if self.f.size() < self.size_end_iter:
                            break
                        if not in_bounds(self.f.xmin, self.bounds):
                            raiseflag(bitmask, i, j, self.L2_FLAG_OUT_OF_BOUNDS)
                            break
            


                # update water model with final parameters
                self.f.w.calc_rho(self.f.xmin)

                logchl[i,j] = self.f.xmin[0]
                eps[i,j] = self.f.fsim[0]
                if self.Nparams >= 2:
                    logfb[i,j] = self.f.xmin[1]
                if self.Nparams >= 3:
                    fa[i,j] = self.f.xmin[2]
                niter[i,j] = self.f.niter
                SPM[i,j] = self.f.w.SPM

                # calculate water reflectance
                # and store atmospheric reflectance
                rw_neg = 0
                for ib in range(self.N_bands_read):
                    Rw[i,j,ib] = Rprime[i,j,ib] - self.f.Ratm[ib]
                    Rw[i,j,ib] /= Tmol[i,j,ib]
                    if Rw[i,j,ib] < 0:
                        rw_neg = 1

                    Rwmod[i,j,ib] = self.f.Rwmod[ib]

                    Ratm[i,j,ib] = self.f.Ratm[ib]
                
                if self.uncertainties:
                    # 1) Uncertainty on the marine parameters
                    # normalize by sigma² = y_min/(N-n), with N = number of observations,
                    # and n = number of parameters fitted
                    # see [Nelder Mead, 1965]
                    sigmasq = self.f.fsim[0]/(self.N_bands_oc-self.Nparams-self.params.Ncoef)
                    self.f.calc_cov(2*sigmasq)

                    logchl_unc[i,j] = self.f.cov[0, 0]
                    logfb_unc[i,j] = self.f.cov[1, 1]

                    # 2) calculate the sensitivity of Rw to the marine parameters
                    for iparam in range(self.Nparams):
                        x0[iparam] = self.f.xmin[iparam]
                    for iparam in range(self.Nparams):
                        x0[iparam] += delta
                        self.f.eval(x0)
                        for ib in range(self.N_bands_read):
                            # the variation of Rw is equal to the opposite of the variation of Ratm
                            d_rw_x[ib, iparam] = (Ratm[i,j,ib] - self.f.Ratm[ib])/delta
                        x0[iparam] = self.f.xmin[iparam]

                    # 3) calculate rho_w_mod_cov from the Jacobian matrix of the model
                    # (eq 55 - 58 of E3UB)
                    # rho_w_mod_cov = d_rw_x . f.cov . d_rw_x'
                    #    [NbxNb]      [NbxNp] [NpxNp] [NpxNb]
                    dot(d_rw_x_cov, d_rw_x, self.f.cov, 0)
                    dot(rho_w_mod_cov, d_rw_x_cov, d_rw_x, 1)

                    for ib in range(self.N_bands_read):
                        rho_w_unc[i,j,ib] = sqrt(rho_w_mod_cov[ib, ib] + Rtoa_var[i,j,ib])/Tmol[i,j,ib]

                    self.f.w.calc_rho(self.f.xmin)

                
                # Store Ci coefficients
                for ib in range(self.Ncoef):
                    Ci[i,j,ib] = self.f.C[ib]
                
                # consistency test at bands_oc
                for ioc in range(self.N_bands_oc):
                    ib = self.i_oc_read[ioc]
                    if (self.Rprime_consistency and (
                              (self.f.Ratm[ib] > Rprime_noglint[i,j,ib])
                           or (self.f.Rwmod[ib]*Tmol[i,j,ib] > Rprime_noglint[i,j,ib]))):
                        raiseflag(bitmask, i, j, self.L2_FLAG_INCONSISTENCY)
                        flag_reinit = 1

                # water reflectance normalization
                if self.normalize:
                    # Rw -> Rw*Rwmod[nadir,lambda0]/Rwmod

                    for ib in range(self.N_bands_read):
                        Rw[i,j,ib] /= self.f.Rwmod[ib]

                    if self.normalize & 1:
                        # activate geometry normalization
                        sza0 = 0.
                        vza0 = 0.
                        raa0 = 0.
                    else:
                        sza0 = sza[i,j]
                        vza0 = vza[i,j]
                        raa0 = raa[i,j]

                    if self.normalize & 2:
                        # activate wavelength normalization
                        wav0 = cwav
                    else:
                        wav0 = wav[i,j,:]

                    # calculate model reflectance at nadir
                    self.f.init_pixel(
                            Rprime[i,j,:],
                            Rprime_noglint[i,j,:],
                            A[i,j,:,:], pA[i,j,:,:],
                            Tmol[i,j,:],
                            wav0,
                            sza0, vza0, raa0,
                            wind_speed[i,j])
                    self.f.w.calc_rho(self.f.xmin)

                    for ib in range(self.N_bands_read):
                        Rw[i,j,ib] *= self.f.Rwmod[ib]

                # thick aerosol flag
                # Rnir/max(Rw) > 10 - 1.5*logchl
                # avoid erroneous retrieval in case of very thick aerosol plumes
                Rw_max = 0.
                for ib in range(self.N_bands_read):
                    if Rw[i,j,ib] > Rw_max:
                        Rw_max = Rw[i,j,ib]
                if (Rnir[i,j]/Rw_max > 10 - 1.5*logchl[i,j]):
                    raiseflag(bitmask, i, j, self.L2_FLAG_THICK_AEROSOL)
                
                # ANOMALY_RWMOD_BLUE flag
                # Removes outliers appearing on MODIS results at high SZA
                # on recent years (eg 2019).
                if Rw[i,j,0] - Rwmod[i,j,0] > 0.005:
                    raiseflag(bitmask, i, j, self.L2_FLAG_ANOMALY_RWMOD_BLUE)

                # initialization of next pixel
                if (self.force_initialization
                        or testflag(bitmask, i, j,  self.L2_FLAG_CASE2)
                        or (rw_neg and self.reinit_rw_neg)
                        or (flag_reinit)
                        ):
                    x0[:] = self.initial_point_1[:]
                    flag_reinit = 0
                else:
                    x0[:] = self.f.xmin[:]


            # reinitialize
            x0[:] = self.initial_point_1[:]

            # check for pending signals
            # (allowing to interrupt execution)
            PyErr_CheckSignals()


    cdef int init_first_guess(self,
                              float[:,:] Rwmod_fg,
                              float[:] cwav):
        """
        Initialize reflectances for first guess

        Rwmod_fg: reflectance spectra [Npts, nbands]
        """
        cdef int i, j
        self.f.w.init_pixel(cwav, 0, 0, 0, 5)
        for i in range(Rwmod_fg.shape[0]):
            self.f.Rwmod = self.f.w.calc_rho(self.initial_points[i,:])
            for j in range(Rwmod_fg.shape[1]):
                Rwmod_fg[i,j] = self.f.Rwmod[j]

        return 0


    cdef first_guess(self,
                     float[:,:] Rwmod_fg, # Spectra calculated for first guess points
                     float[:] x0,
                     int i, int j):
        cdef float v_fguess, vmin_fguess
        cdef int i_fguess=0, ii
        vmin_fguess = -1
        v_fguess = -1
        for ii in range(self.initial_points.shape[0]):
            if self.firstguess_method == 0:
                # old method
                v_fguess = self.f.eval(self.initial_points[ii,:])
            else:
                # new method
                # avoid calling the water model each time, by
                # using the pre-calculated Rwmod_fg
                for k in range(Rwmod_fg.shape[1]):
                    self.f.Rwmod[k] = Rwmod_fg[ii,k]
                v_fguess = self.f.eval_atm(self.initial_points[ii,:])

            if (vmin_fguess < 0) or (v_fguess < vmin_fguess):
                vmin_fguess = v_fguess
                i_fguess = ii
            
        # Include last point in first guess
        # => With current values of self.f.Rwmod and self.f.xmin
        if ((not self.force_initialization)
            and not isnan(self.f.xmin[0])
            and in_bounds(self.f.xmin, self.bounds)
            and (self.f.eval_atm(self.f.xmin) < v_fguess)):

            # Reuse previous pixel (if better than all first guess pixels)
            for ii in range(x0.shape[0]):
                x0[ii] = self.f.xmin[ii]
        else:
            # Use first guess value
            for ii in range(x0.shape[0]):
                x0[ii] = self.initial_points[i_fguess,ii]

        if self.dbg_pt[0] >= 0:
            if ((self.dbg_pt[0] == i) and (self.dbg_pt[1] == j)):
                print('first guess: selected [{}] : ({}, {})'.format(i_fguess, x0[0], x0[1]))


    def minimize(self, block):
        '''
        Call minimization code for a block
        (def method visible from python code)
        '''

        # calculate glint reflectance from wind speed
        ok = (block.bitmask & self.BITMASK_INVALID) == 0
        block.Rgli = np.zeros_like(block.wind_speed, dtype='float32') + np.nan
        block.Rgli[ok] = glitter(block.wind_speed[ok],
                                 block.mus[ok], block.muv[ok],
                                 block.scattering_angle[ok], phi=None, phi_vent=None)

        if self.params.partial >= 1:
            return

        # calculate the atmospheric inversion coefficients
        # at bands_corr
        A = atm_func(block, self.params, self.params.bands_corr)
        if self.params.weights_corr is None:
            pA = pseudoinverse(A)
        else:
            pA = weighted_pseudoinverse(
                    A, np.diag(self.params.weights_corr).astype('float32'))

        # the model coefficients, at bands_read
        A = atm_func(block, self.params, self.params.bands_read())

        self.loop(block, A, pA)

    def visu_costfunction(self):
        '''
        Visualization of the cost function for current pixel
        '''
        from matplotlib.pyplot import pcolor, show, colorbar, plot

        NX, NY = 100, 100
        cost = np.zeros((NX, NY), dtype='float32')
        tab_p = np.array(np.meshgrid(
            np.linspace(-2, 0, NX),
            np.linspace(-0.5, 0.5, NY)), dtype='float32')
        for i in range(NX):
            for j in range(NY):
                cost[i,j] = self.f.eval(tab_p[:,i,j])

        pcolor(tab_p[0,:,:], tab_p[1,:,:],
               np.log10(cost), cmap='coolwarm')
        colorbar()

        # plot iterations
        xprev = None
        # self.f.init(self.initial_point_1, self.initial_step)
        xprev = self.initial_point_1
        while self.f.niter < self.max_iter:

            self.f.iterate()

            plot([xprev[0], self.f.xmin[0]],
                 [xprev[1], self.f.xmin[1]],
                 'k-')
            xprev = list(self.f.xmin)

            if self.f.size() < self.size_end_iter:
                break

