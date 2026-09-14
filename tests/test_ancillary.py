#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from datetime import datetime, timedelta
from polymer.ancillary import Ancillary_NASA
from polymer.ancillary_era5 import Ancillary_ERA5, resolve_expver
from matplotlib import pyplot as plt
from . import conftest
from os import system
from tempfile import TemporaryDirectory


def _expver_dataset(final_values, preliminary_values):
    '''
    Build a synthetic dataset shaped like CDS's ERA5 response when an
    `expver` dimension is present, with a single dummy 2x2 variable 'sp'.
    Pass None for a given branch to omit that expver value entirely.
    '''
    expvers = []
    data = []
    for expver, values in [(1, final_values), (5, preliminary_values)]:
        if values is not None:
            expvers.append(expver)
            data.append(np.full((2, 2), values, dtype=float))

    return xr.Dataset(
        {'sp': (('expver', 'y', 'x'), np.stack(data))},
        coords={'expver': expvers},
    )


def test_resolve_expver_no_expver_dim():
    ds = xr.Dataset({'sp': (('y', 'x'), np.zeros((2, 2)))})
    ds_out, source = resolve_expver(ds)
    assert source == 'final'
    assert ds_out is ds


def test_resolve_expver_final_available():
    ds = _expver_dataset(final_values=1013., preliminary_values=1012.)
    ds_out, source = resolve_expver(ds)
    assert source == 'final'
    assert (ds_out.sp.values == 1013.).all()


def test_resolve_expver_falls_back_to_preliminary():
    ds = _expver_dataset(final_values=np.nan, preliminary_values=1012.)
    ds_out, source = resolve_expver(ds, allow_preliminary=True)
    assert source == 'preliminary'
    assert (ds_out.sp.values == 1012.).all()


def test_resolve_expver_preliminary_disallowed():
    ds = _expver_dataset(final_values=np.nan, preliminary_values=1012.)
    with pytest.raises(Exception):
        resolve_expver(ds, allow_preliminary=False)


def test_resolve_expver_no_data_at_all():
    ds = _expver_dataset(final_values=np.nan, preliminary_values=np.nan)
    with pytest.raises(Exception):
        resolve_expver(ds)


@pytest.mark.parametrize('variable,typ_value', [
    ('wind_speed', 10),
    ('surf_press', 1013),
    ('ozone', 400.),
])
@pytest.mark.parametrize('mode,offset', [  # offset=number of days
    ('NASA', 1),
    ('NASA', 20),
    ('NASA', 100),
    ('ERA5', 1),  # exercises the ERA5T/expver fallback path
    ('ERA5', 20),
    ('ERA5', 100),
])
def test_ancillary(request, variable, typ_value, mode, offset):
    with TemporaryDirectory() as tmpdir:
        if mode == 'NASA':
            anc = Ancillary_NASA(directory=tmpdir)
        elif mode == 'ERA5':
            anc = Ancillary_ERA5(directory=tmpdir)
        else :
            raise ValueError(mode)
            
        ret = anc.get(variable, datetime.now() - timedelta(days=offset))
        print(ret)
        print(ret.date)
        print(ret.filename)

        assert ret.data.data.mean() < typ_value*1.5
        assert ret.data.data.mean() > typ_value*0.5

        plt.figure()
        plt.imshow(ret.data.data)
        plt.colorbar()
        conftest.savefig(request)


@pytest.mark.parametrize('url',[
    'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/GMAO_FP.20231005T090000.MET.NRT.nc', # Available file
    ])
def test_download(url):
    with TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir)/Path(url).name
        ret = Ancillary_NASA().download(url, str(tmpfile))
        print(ret)
        assert ret == 0

@pytest.mark.parametrize('url',[
    'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/25061439.nc', # 404 Error
    # 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/N202000300_O3_AURAOMI_24h.hdf'     , # 403 Error
    ])
def test_download_nofile(url):
    with TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir)/Path(url).name
        ret = Ancillary_NASA().download(url, str(tmpfile))
        assert ret != 0

def test_download_auth(tmp_path):
    """Test that wget download with NASA auth cookies works."""
    url = 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/GMAO_FP.20231005T090000.MET.NRT.nc'
    tmpfile = tmp_path / 'test_auth.tmp'
    cmd = 'wget -nv --save-cookies ~/.urs_cookies --keep-session-cookies --auth-no-challenge {} -O {}'.format(url, tmpfile)
    assert system(cmd) == 0
    assert tmpfile.exists()