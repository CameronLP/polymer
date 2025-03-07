import numpy as np
cimport numpy as np
from cython cimport floating
from libc.math cimport abs, sqrt, nan
import sys


cdef class NelderMeadMinimizer:

    def __init__(self, int N):
        '''
        Initialize the minimizer with a number of dimensions N
        '''
        self.N = N
        self.fsim = np.zeros((N + 1,), dtype='float32') + np.nan
        self.sim = np.zeros((N + 1, N), dtype='float32')
        self.ssim = np.zeros((N + 1, N), dtype='float32')
        self.xbar = np.zeros(N, dtype='float32')
        self.ind = np.zeros((N + 1,), dtype='int32')
        self.y = np.zeros(N, dtype='float32')
        self.xmin = np.zeros(N, dtype='float32') + np.nan
        self.xcc = np.zeros(N, dtype='float32')
        self.xc = np.zeros(N, dtype='float32')
        self.xr = np.zeros(N, dtype='float32')
        self.xe = np.zeros(N, dtype='float32')
        self.center = np.zeros(N, dtype='float32')

        # covariance matyix calculation
        self.cov = np.zeros((N, N), dtype='float32') + np.nan
        self.B = np.zeros((N, N), dtype='float32') + np.nan
        self.Binv = np.zeros((N, N), dtype='float32') + np.nan
        self.fmid = np.zeros((N+1, N+1), dtype='float32') + np.nan
        self.Q = np.zeros((N, N), dtype='float32') + np.nan
        self.Q_Binv = np.zeros((N, N), dtype='float32') + np.nan

    cdef float eval(self, float[:] x):
        raise Exception('NelderMeadMinimizer.eval() shall be implemented')

    cdef float size(self):
        '''
        calculate the simplex size as average lengths of vectors from center xbar to corners
        '''
        cdef int i, j
        cdef float val, dx, s

        # calculate center
        for j in range(self.N):
            val = 0.
            for i in range(self.N+1):
                val += self.sim[i,j]
            self.center[j] = val/(self.N+1)

        # calculate size
        s = 0.
        for i in range(self.N+1):
            # for each N+1 corner
            val = 0.
            for j in range(self.N):
                dx = self.ssim[i,j] - self.center[j]
                dx *= dx
                val += dx

            s += sqrt(val)

        return s/(self.N+1)

    cdef int init(self,
            float[:] x0,
            float[:] dx,
            ):
        '''
        Initialize the Nelder-Mead minimize with initial vector x0 and initial
        step dx
        '''
        self.niter = 0
        if self.N != x0.shape[0]:
            raise Exception('')
        cdef int N = self.N
        cdef float[:] y = self.y
        cdef int k, j

        for j in range(N):
            self.sim[0,j] = x0[j]
        self.fsim[0] = self.eval(x0)
        for k in range(N):
            for j in range(N):
                y[j] = x0[j]
            y[k] += dx[k]

            for j in range(N):
                self.sim[k + 1, j] = y[j]
            self.fsim[k + 1] = self.eval(y)

        combsort(self.fsim, self.N+1, self.ind)

        # use indices to sort the simulation parameters
        for k in range(self.N+1):
            for j in range(N):
                self.ssim[k,j] = self.sim[self.ind[k],j]
        for k in range(self.N+1):
            for j in range(N):
                self.sim[k,j] = self.ssim[k,j]
        
        return 0


    cdef int iterate(self):
        cdef int N = self.N
        cdef float[:] y = self.y
        cdef float fxr, fxe, fxc, fxcc
        cdef int k, j

        self.niter += 1

        # calculate centroid of all points but last
        for j in range(N):
            self.xbar[j] = 0.
            for k in range(self.N):
                self.xbar[j] += self.sim[k,j]
            self.xbar[j] /= N

        # reflection
        for k in range(N):
            self.xr[k] = 2*self.xbar[k] - self.sim[-1,k]
        fxr = self.eval(self.xr)
        doshrink = 0

        if fxr < self.fsim[0]:
            # the reflected point is the best so far:
            # expand
            for k in range(N):
                self.xe[k] = 3*self.xbar[k] - 2*self.sim[-1,k]
            fxe = self.eval(self.xe)

            if fxe < fxr:
                for k in range(N):
                    self.sim[N,k] = self.xe[k]
                self.fsim[N] = fxe
            else:
                for k in range(N):
                    self.sim[N,k] = self.xr[k]
                self.fsim[N] = fxr
        else:  # fsim[0] <= fxr
            if fxr < self.fsim[N-1]:
                for k in range(N):
                    self.sim[N,k] = self.xr[k]
                self.fsim[N] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < self.fsim[N]:
                    for k in range(N):
                        self.xc[k] = 1.5 * self.xbar[k] - 0.5 * self.sim[-1,k]
                    fxc = self.eval(self.xc)

                    if fxc <= fxr:
                        for k in range(N):
                            self.sim[N, k] = self.xc[k]
                        self.fsim[N] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    for k in range(N):
                        self.xcc[k] = 0.5*(self.xbar[k] + self.sim[-1,k])
                    fxcc = self.eval(self.xcc)

                    if fxcc < self.fsim[N]:
                        for k in range(N):
                            self.sim[N,k] = self.xcc[k]
                        self.fsim[N] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in range(1, N+1):
                        for k in range(N):
                            self.sim[j,k] = self.sim[0,k] + 0.5 * (self.sim[j,k] - self.sim[0,k])
                            y[k] = self.sim[j,k]
                        self.fsim[j] = self.eval(y)

        combsort(self.fsim, self.N+1, self.ind)
        # use indices to sort the simulation parameters
        for k in range(self.N+1):
            for j in range(N):
                self.ssim[k,j] = self.sim[self.ind[k],j]
        for k in range(self.N+1):
            for j in range(N):
                self.sim[k,j] = self.ssim[k,j]

        for j in range(self.N):
            self.xmin[j] = self.sim[0,j]

    cdef float[:] minimize(self,
                float[:] x0,
                float[:] dx,
                float size_end_iter,
                int maxiter=-1):
        """
        Minimization of scalar function of one or more variables using the
        Nelder-Mead algorithm.
        """
        if maxiter < 0:
            maxiter = self.N * 200

        self.init(x0, dx)


        while self.niter < maxiter:

            if self.size() < size_end_iter:
                break

            self.iterate()



        return self.xmin
    
    cdef calc_cov(self, float coef):
        """
        Calculate the variance-covariance matrix at the minimum
        see Nelder and Mead, 1965, appendix
        (https://people.duke.edu/~hpgavin/cee201/Nelder+Mead-ComputerJournal-1965.pdf)

        The simplex points are sim[N+1,N], and the corresponding function values are fsim[N+1]

        `coef` is a normalization coefficient that is multiplied to the final matrix
        """
        cdef int i, j, k
        cdef int reset_cov = 0

        # calculate the midpoint values (fmid)
        for i in range(self.N+1):
            for j in range(self.N+1):
                if i >= j:
                    continue
                for k in range(self.N):
                    self.y[k] = 0.5*(self.sim[i,k] + self.sim[j,k])
                self.fmid[i,j] = self.eval(self.y)
        
        # transfer matrix Q
        for i in range(self.N):
            for k in range(self.N):
                self.Q[i,k] = self.sim[i+1,k] - self.sim[0,k]

        # calculate the quadratic approximation
        # (only matrix B is necessary)
        for i in range(self.N):
            # diagonal terms
            self.B[i,i] = 2*(self.fsim[i+1] + self.fsim[0]
                             - 2*self.fmid[0,i+1])
            for j in range(self.N):
                # off-diagonal terms
                if i >= j:
                    continue
                self.B[i,j] = 2*(
                    self.fmid[i+1,j+1]
                    + self.fsim[0]
                    - self.fmid[0,i+1]
                    - self.fmid[0,j+1])
                self.B[j,i] = self.B[i,j]
        
        # Calculate the variance-covariance matrix
        invert(self.Binv, self.B)

        # cov = Q.B^-1.Q'
        dot(self.Q_Binv, self.Q, self.Binv, 0)
        dot(self.cov, self.Q_Binv, self.Q, 1)

        # Apply normalization coefficient
        for i in range(self.N):
            for j in range(self.N):
                self.cov[i,j] *= coef
            if self.cov[i,i] < 0:
                reset_cov = 1

        if reset_cov:
            for i in range(self.N):
                for j in range(self.N):
                    self.cov[i,j] = 0.


cdef invert(float[:,:] Ainv, float[:,:] A):
    """
    Invert matrix A to Ainv
    """
    if (A.shape[0] != 2) or A.shape[1] != 2:
        print('Error in neldermead.invert')
        sys.exit(1)
    
    cdef float det = A[0,0]*A[1,1] - A[1,0]*A[0,1]
    Ainv[0,0] = A[1,1]/det
    Ainv[1,1] = A[0,0]/det
    Ainv[0,1] = -A[1,0]/det
    Ainv[1,0] = -A[0,1]/det


cdef dot(float[:,:] C, float[:,:] A, float[:,:] B, int transpose_B):
    """
    Matrix product C = A.B
    (or C=A.B')
    """
    cdef int i, j, k
    if not transpose_B:
        if ((C.shape[0] != A.shape[0]) or (A.shape[1] != B.shape[0]) or (C.shape[1] != B.shape[1])):
            print('Shape error in neldermead.dot')
            sys.exit(1)
    else:
        if ((C.shape[0] != A.shape[0]) or (A.shape[1] != B.shape[1]) or (C.shape[1] != B.shape[0])):
            print('Shape error in neldermead.dot')
            sys.exit(1)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i, j] = 0
            for k in range(A.shape[1]):
                if not transpose_B:
                    C[i, j] += A[i,k]*B[k,j]
                else:
                    C[i, j] += A[i,k]*B[j,k]

cdef int combsort(float[:] inp, int N, int[:] ind):
    '''
    in-place sort of array inp of size N using comb sort.
    returns sorting indexes in array ind.
    '''
    cdef int gap = N
    cdef int swapped = 0
    cdef float shrink = 1.3
    cdef int i
    cdef int ix
    cdef float tmp
    for i in range(N):
        ind[i] = i

    while not ((gap == 1) and (not swapped)):
        gap = int(gap/shrink)
        if gap < 1:
            gap = 1
        i = 0
        swapped = 0

        while i + gap < N:

            if inp[i] > inp[i+gap]:

                # swap the values in place
                tmp = inp[i+gap]
                inp[i+gap] = inp[i]
                inp[i] = tmp

                # swap also the index
                ix = ind[i+gap]
                ind[i+gap] = ind[i]
                ind[i] = ix

                swapped = 1

            i += 1

    return 0

def test_combsort():
    N = 10
    A = np.random.randn(N).astype('float32')
    AA = A.copy()
    I = np.zeros(N, dtype='int32')
    combsort(A, N, I)
    assert (np.diff(A) >= 0).all()
    assert (AA[I] == A).all()


cdef class Rosenbrock(NelderMeadMinimizer):
    cdef float eval(self, float[:] x):
        # rosenbrock function
        return (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])

cdef test_minimize():
    r = Rosenbrock(2)
    DX = np.array([0.1, 0.1], dtype='float32')
    for X0 in [
            np.array([0, 0], dtype='float32'),
            np.array([-1, -1], dtype='float32'),
            # np.array([0, -1], dtype='float32'),
            # np.array([10, 0], dtype='float32'),
            # np.array([0, 10], dtype='float32'),
            ]:

        X = np.array(r.minimize(X0, DX, 0.001))
        assert r.niter > 10
        assert (np.abs(X - 1) < 0.01).all(), (X0, X)



def test():
    '''
    module-wise testing
    '''
    test_combsort()
    test_minimize()

