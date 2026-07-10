from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xarray as xr
from core.tools import Var
from core.network.download import download_url
from core.env import getdir
from core.process.blockwise import BlockProcessor, CompoundProcessor
from core.interpolate import interp, Linear
from eotools.solar_irradiance import solar_irradiance_lisird
from eotools.units import convert

"""
Definition of top of atmosphere uncertainties
"""

vardef = Var('Rtoa_var', dtype="float32", dims=('y', 'x', 'bands'))

class InitUncertainties(CompoundProcessor):
    """
    Initialize the input uncertainties (Rtoa_var) from TOA radiance.
    Computes F0 (solar irradiance) if not present, then applies
    sensor-appropriate TOA uncertainty model.
    """
    def __init__(self, ds: xr.Dataset, params):
        self.ds = ds
        self.params = params
        
        # Compute F0 (solar irradiance) if not already present
        # F0 is a dataset-level 1D variable (bands only), added directly to ds
        if 'F0' not in ds:
            solar_data = solar_irradiance_lisird("1nm")
            F0 = solar_data.SSI.compute(scheduler='sync')
            ds["F0"] = interp(F0, wavelength=Linear(ds.cwav))

        # Build processor chain
        processors: list[BlockProcessor] = []

        # Only compute uncertainties if enabled in params
        if params.uncertainties:
            processors.append(Init_Ltoa(ds))

            if ("product_name" in ds.attrs) and (ds.attrs['product_name'].startswith('PACE_OCI')):
                processors.append(TOA_Uncertainties_PACE(ds))
            else:
                processors.append(TOA_Uncertainties(ds, params))

        super().__init__(processors)


class Init_Ltoa(BlockProcessor):
    def __init__(self, ds: xr.Dataset):
        """
        Add TOP of atmosphere irradiance if not present already
        """
        self.activate = 'Ltoa' not in ds
    
    def input_vars(self) -> list[Var]:
        return [Var('Rtoa'), Var('mus'), Var('F0')]
    
    def created_vars(self) -> list[Var]:
        if self.activate:
            return [Var("Ltoa", dtype="float32", dims_like="Rtoa")]
        else:
            return []
    
    def process_block(self, block: xr.Dataset) -> None:
        Ltoa = (1 / np.pi) * block.mus * block.F0 * block.Rtoa
        block['Ltoa'] = Ltoa.astype('float32').transpose(*block.Rtoa.dims)
        
        assert block.F0.units == "W m-2 nm-1"
        block['Ltoa'].attrs.update(units='W m-2 sr-1 nm-1')


class TOA_Uncertainties(BlockProcessor):
    def __init__(self, ds: xr.Dataset, params):
        self.Ltyp = xr.DataArray(
                data=list(params.Ltyp.values()),
                dims=['bands'],
                coords={'bands': list(params.Ltyp)})
        self.sigma_typ = xr.DataArray(
                data=list(params.sigma_typ.values()),
                dims=['bands'],
                coords={'bands': list(params.sigma_typ)})
        
    def input_vars(self) -> list[Var]:
        return [
            Var("Ltoa"),
            Var("F0"),
            Var("mus"),
        ]
    
    def created_vars(self) -> list[Var]:
        return [Var('Rtoa_var')]
    
    def auto_template(self) -> bool:
        return True
    
    def process_block(self, block: xr.Dataset):
        Rtoa_var = (block.Ltoa/self.Ltyp) * (np.pi*self.sigma_typ/(block.F0*block.mus))**2
        block['Rtoa_var'] = Rtoa_var.astype('float32')


class TOA_Uncertainties_PACE(BlockProcessor):
    def __init__(self, ds: xr.Dataset):
        # Load PACE uncerainty model

        url = 'https://oceancolor.gsfc.nasa.gov/images/data/PACE_OCI_L1B_LUT_baseline_SNR_1.1.txt'
        f = download_url(url, dirname=getdir('DIR_STATIC')/'PACE_OCI')

        # Find the start of data after /end_header
        with open(f, 'r') as file:
            lines = file.readlines()
        start_line = 0
        for i, line in enumerate(lines):
            if line.startswith('/end_header'):
                start_line = i + 1
                break

        # Read the data with pandas
        data_snr = pd.read_csv(
            f,
            sep=r'\s+',  # Multiple whitespaces as separator
            skiprows=start_line,
            header=None,
            names=["FPA", "wavelength", "band_index", "c1", "c2"],
        ).to_xarray()
        data_snr = data_snr.assign_coords(index=data_snr.wavelength).rename(index='wav')
        data_snr = data_snr.sortby('wav')

        # Interpolate c1 and c2 to the dataset wavelengths
        cwav = ds.cwav.compute()
        self.c1 = interp(data_snr['c1'], wav=Linear(cwav))
        self.c2 = interp(data_snr['c2'], wav=Linear(cwav))
        
    def input_vars(self) -> list[Var]:
        return [Var('Ltoa')]

    def created_vars(self) -> list[Var]:
        return [Var('Rtoa_var', dtype='float32', dims_like='Ltoa')]

    def initialize(self, ds: xr.Dataset) -> xr.Dataset:
        return ds

    def process_block(self, block: xr.Dataset):
        # Convert Ltoa to W/m^2/sr/um as required by c1 and c2
        Ltoa = convert(block.Ltoa, "W/m^2/sr/um")
        Rtoa_var = (self.c1 + self.c2 * Ltoa) / Ltoa**2
        block["Rtoa_var"] = Rtoa_var.astype('float32')


def toa_uncertainties(block, dir_common):
    """
    Note: this function is either called with the v4 block structure, or with the
        v5 Dataset structure.
    """
    if not isinstance(block, xr.Dataset):
        mus = block.mus[...,None]
        cwav = block.cwavelen
    else:
        mus = block.mus
        cwav = block.cwav

    if hasattr(block, 'F0'):
        F0 = block.F0
    else:
        # Interpolation of F0 in solar spectrum file
        solar_spectrum_file = Path(dir_common)/'SOLAR_SPECTRUM_WMO_86'
        solar_data = pd.read_csv(solar_spectrum_file, sep=' ')

        F0_interp = interp1d(solar_data['lambda(nm)'], solar_data['Sl(W.m-2.nm-1)'])
        F0 = F0_interp(cwav)*1000 # interpolate and convert to µm-1
        if isinstance(block, xr.Dataset):
            F0 = xr.DataArray(F0, dims='bands')


    if hasattr(block, 'Ltoa'):
        Ltoa = block.Ltoa
    else:
        assert hasattr(block, 'Rtoa')
        Ltoa = (1/np.pi)*mus*F0*block.Rtoa

    Rtoa_var = (Ltoa/block.Ltyp) * (np.pi*block.sigma_typ/(F0*mus))**2

    if isinstance(block, xr.Dataset):
        return vardef.conform(Rtoa_var.astype('float32'), transpose=True)
    else:
        return Rtoa_var.astype('float32')
