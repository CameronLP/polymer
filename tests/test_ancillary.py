#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
from datetime import datetime, timedelta
from polymer.ancillary import Ancillary_NASA
from polymer.ancillary_era5 import Ancillary_ERA5
from matplotlib import pyplot as plt
from . import conftest
from os import system
from tempfile import TemporaryDirectory


@pytest.mark.parametrize('variable,typ_value', [
    ('wind_speed', 10),
    ('surf_press', 1013),
    ('ozone', 400.),
])
@pytest.mark.parametrize('mode,offset', [  # offset=number of days
    ('NASA', 1),
    ('NASA', 20),
    ('NASA', 100),
    # ('ERA5', 1),  # expected to fail (data not yet available)
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