#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
import tempfile
from matplotlib import pyplot as plt
from polymer.level1_msi import Level1_MSI
from polymer.level2_nc import Level2_NETCDF
from polymer import level2
from polymer.main import run_atm_corr
from eoread import eo
import xarray as xr
from . import conftest
from .test_samples import sample


@pytest.fixture
def msi_product() -> Path:
    return sample('LEVEL1_SAMPLE_MSI')

def test_instantiate(msi_product):
    print(Level1_MSI(msi_product))
    

@pytest.mark.parametrize('uncertainties', [True, False])
def test_msi(request, msi_product, uncertainties):

    l1 = Level1_MSI(
            msi_product,
            ancillary='ECMWFT',
            sline=500, eline=800,
            scol=1000, ecol=1400, resolution='60')
    with tempfile.TemporaryDirectory() as tmpdir:
        ret = run_atm_corr(
            l1,
            Level2_NETCDF(outdir=tmpdir,
                          datasets=(level2.default_datasets+level2.uncertainty_datasets) if uncertainties else level2.default_datasets),
            uncertainties=uncertainties,
        )
        print('Created file:', ret)
        ds = xr.open_dataset(ret.filename)
        assert 'Rgli' in ds

        plt.figure()
        plt.imshow(ds.rho_w_unc490 if uncertainties else ds.Rw490)
        plt.colorbar()

        conftest.savefig(request)


def test_msi_spectrum(request, msi_product):
    l1 = Level1_MSI(
            msi_product,
            sline=500, eline=510,
            scol=1000, ecol=1010)
    with tempfile.TemporaryDirectory() as tmpdir:

        l2 = run_atm_corr(l1, Level2_NETCDF(outdir=tmpdir, datasets=['Rw', 'Ratm', 'Rprime']))

        l2 = xr.open_dataset(l2.filename)
        l2 = eo.merge(l2, dim='wav', pattern=r'Rprime(\d+)', varname='Rprime')
        l2 = eo.merge(l2, dim='wav', pattern=r'Ratm(\d+)', varname='Ratm')
        l2 = eo.merge(l2, dim='wav', pattern=r'Rw(\d+)', varname='Rw')
        l2 = l2.isel(height=0, width=0)
        l2.Rw.plot(label='Rw')
        l2.Ratm.plot(label='Ratm')
        l2.Rprime.plot(label='Rprime')
        plt.grid(True)
        plt.legend()
        conftest.savefig(request)



