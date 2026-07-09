from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr
from core.env import getdir
from core.tests.conftest import savefig
from polymer.water import ParkRuddick
from polymer.main_v5 import normalize_water_reflectance


def test_normalize(request):
    """
    Test the use of Polymer's main function for BRDF correction
    with 3 different chlorophyll concentrations along dimension "x".
    """
    # Generate sample water reflectance spectrum
    # 10 bands evenly spaced from 400 to 850 nm
    wav = np.linspace(400, 850, 10).astype('float32')
    bands = np.round(wav).astype(int).tolist()

    # Three water types along dimension "x"
    # logchl: log10(chlorophyll concentration in mg/m³)
    logchl_vals = np.array([-1.0, 0.0, 1.0], dtype='float32')
    logfb = 0.0   # log10(backscattering factor)
    logfa = 0.0   # log10(CDM absorption factor)

    # Geometry & environment (same for all pixels)
    sza = 30.0    # solar zenith angle (degrees)
    vza = 60.0    # viewer zenith angle (degrees)
    raa = 0.0     # relative azimuth angle (degrees)
    ws = 5.0      # wind speed (m/s)

    # Initialize water model
    dir_static = getdir("DIR_POLYMER_AUXDATA",
                         getdir("DIR_DATA", Path("auxdata")) / "static")
    AUXDATA_COMMON = str(dir_static / "common")
    model = ParkRuddick(AUXDATA_COMMON, absorption='bricaud98_aphy', bbopt=0)

    # Calculate water reflectance ρ_w for each chlorophyll concentration
    num_chl = len(logchl_vals)
    rho_w_stack = np.zeros((1, num_chl, len(wav)), dtype='float32')
    for i, logchl in enumerate(logchl_vals):
        rho_w_stack[0, i, :] = model.calc(
            w=wav,
            logchl=logchl,
            logfb=logfb,
            logfa=logfa,
            sza=sza,
            vza=vza,
            raa=raa,
            ws=ws,
        )

    # Build xarray Dataset with 3 pixels along "x"
    ds = xr.Dataset({
        'Rprime': xr.DataArray(
            rho_w_stack,
            dims=['y', 'x', 'bands'],
        ),
        'wav': xr.DataArray(wav, dims=['bands']),
        'sza': xr.DataArray([[sza] * num_chl], dims=['y', 'x']).astype('float32'),
        'vza': xr.DataArray([[vza] * num_chl], dims=['y', 'x']).astype('float32'),
        'raa': xr.DataArray([[raa] * num_chl], dims=['y', 'x']).astype('float32'),
    })
    # Assign integer band labels and x coordinates
    ds = ds.assign_coords({'bands': bands, 'x': logchl_vals})

    # Run PolymerSolver via normalize_water_reflectance (single call)
    ds = normalize_water_reflectance(ds)

    # Plot original vs. normalized rho_w for each chlorophyll concentration
    labels = ['logchl=-1', 'logchl=0', 'logchl=1']

    for i, logchl in enumerate(logchl_vals):
        rho_w_orig = rho_w_stack[0, i, :]
        rho_w_norm = ds['rho_w'].values[0, i, :]

        plt.plot(wav, rho_w_orig, 'o-', label=f'Original [{labels[i]}]', color=f'C{i}')
        plt.plot(wav, rho_w_norm, 's--', label=f'Normalized [{labels[i]}]', color=f'C{i}')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel(r'$\rho_w$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    savefig(request)



