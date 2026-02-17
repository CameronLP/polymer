#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import xarray as xr
from core.env import getvar
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.make_L1C import makeL1C
from eoread.nasa import Level1_NASA
from eotools import srf
from matplotlib import pyplot as plt

from polymer.main_v5 import run_polymer_dataset
from tests.conftest import savefig

level1A_hawkeye = {
    # SEAHAWK1_HAWKEYE.20230701T160442.L1A.nc
    "path": Path(getvar("LEVEL1A_SAMPLE_HAWKEYE")),
    "band_nir": 867,
    "poi": {"x": 100, "y": 300},   # Within roi
    "roi": {"x": slice(0, 500), "y": slice(4700, 5200)},
}

def test_hawkeye(request):
    l1 = Level1_NASA(makeL1C(level1A_hawkeye["path"])).isel(level1A_hawkeye["roi"])
    l2 = run_polymer_dataset(l1).compute()
    with xr.set_options(display_max_rows=100):
        print(l2)
    
    # Plot area
    l2.rho_w.sel(bands=488).plot(vmin=0, vmax=0.05)
    savefig(request)

    # Plot point of interest
    for varname in ['Ratm', 'rho_w', 'Rprime']:
        l2[varname].sel(level1A_hawkeye['poi']).plot()
    plt.legend()
    plt.grid(True)
    savefig(request)




    


    