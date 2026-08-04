import os
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import xarray as xr
from core.save import to_netcdf
from core.tools import split
from core.process.blockwise import BlockProcessor, CompoundProcessor
from core.tools import Var
from dask import config as dask_config
from dask.distributed import Client
from eoread.autodetect import Level1
from eoread.eo import init_Rtoa
from eoread.flags import FlagsInit, GenericFlags
from eotools.apply_ancillary import ApplyAncillary
from eotools.cm.basic import Cloud_mask
from eotools.gaseous_correction import Gaseous_correction
from eotools.glint import CalcSunGlint
from eotools.rayleigh import RayleighCorrection
from eotools.srf import get_SRF, integrate_srf, rename
from eotools.water import GSWLandMask
from eotools.dem import DEM
from eotools.geometry import InitGeometry

from polymer.common import L2FLAGS
from polymer.polymer_solver import PolymerSolver
from polymer.uncertainties import InitUncertainties
from polymer.water import ParkRuddick
from polymer.params import Params


default_output_datasets = [
    "latitude",
    "longitude",
    "rho_w",
    "logchl",
    "logfb",
    "Rgli",
    "Rnir",
    "flags",
]

additional_output_datasets = [
    "Rtoa",
    "rho_gc",
    "Rprime",
    "Ratm",
]


def run_polymer(
    level1: Union[Path, str, xr.Dataset],
    *,
    roi: Optional[Dict] = None,
    chunks: Optional[Dict] | int = None,
    file_out: Optional[Path | str] = None,
    ext: str = ".polymer.nc",
    dir_out: Optional[Path | str] = None,
    multiprocessing: int | object = 0,
    split_bands: bool = True,
    if_exists: Literal["skip", "overwrite", "backup", "error"] = "error",
    verbose: bool = True,
    outputs: Literal["created_modified", "all", "tags", "named"] = "tags",
    outputs_tags: Optional[list[str]] = ["level2"],
    outputs_names: Optional[list[str]] = None,
    **kwargs,
) -> Path:
    """
    Polymer: main function at file level

    Arguments:
        level1 is either a Path or a xr.Dataset (read with the eoread library)
        roi: definition of the region of interest. A dictionary such as
             {'x': slice(xmin, xmax, xstep), # or [xmin, xmax, xstep]
              'y': slice(ymin, ymax, ystep)}
        chunks: chunking configuration to pass to the Level1 constructor.
                Example: {'y': 200, 'x': 200}
        file_out (Path, optional): path to the output file. If not provided, use the
            two next arguments to determine the output file.
        ext (str): output filename extension
        dir_out (Path, optional): path to the output directory
        multiprocessing: dask execution mode
            - 0: single-threaded, synchronous execution (default)
            - -1: use dask.distributed Client with all available CPU cores
            - N (int > 0): use dask.distributed Client with N workers
            - <other>: pass-through value for dask scheduler (e.g., "threads", "processes")
        split_bands (bool): whether to split the output spectral bands into individual
            variables. Example: rho_w -> [rho_w_412, rho_w_443, ...]
        output_datasets: list of datasets to write to the output product.
            In case of empty list, print all available datasets and exit.
        if_exists: how to deal with existing output file
            ["skip", "overwrite", "backup", "error"]

    Returns:
        The path to the output product.
    """
    if isinstance(level1, (Path, str)):
        kw = {"chunks": chunks} if chunks is not None else {}
        ds = Level1(Path(level1), **kw)
        basename = Path(level1).name
    elif isinstance(level1, xr.Dataset):
        ds = level1
        basename : str = ds.attrs['product_name']
    else:
        raise TypeError('Error in level1 dtype')

    if file_out is None:
        # determine file_out from dir_out and ext
        assert dir_out is not None
        file_out = Path(dir_out) / (basename + ext)
    assert file_out is not None
    
    if (roi is not None):
        ds = ds.isel({k: (v if isinstance(v, slice)
                      else slice(*v))
                      for k, v in roi.items()})

    # Run polymer main function
    ds = run_polymer_dataset(
        ds,
        outputs=outputs,
        outputs_tags=outputs_tags,
        outputs_names=outputs_names,
        **kwargs,
    )

    if split_bands:
        ds = split(ds, 'bands')

    if isinstance(multiprocessing, int):
        if multiprocessing == 0:
            scheduler = "sync"
        elif multiprocessing == -1:
            scheduler = Client(n_workers=os.cpu_count())
        else:
            scheduler = Client(n_workers=multiprocessing)
    else:
        scheduler = multiprocessing

    with dask_config.set(scheduler=scheduler):
        to_netcdf(ds, filename=Path(file_out), if_exists=if_exists, verbose=verbose)
    
    return Path(file_out)


def init(
    ds: xr.Dataset, srf: xr.Dataset | None, params
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    Initialize dataset `ds` for use with Polymer
    (in place)
    
    Returns:
        Tuple of (ds, srf) with bands renamed if bands_l1 is defined.
    """
    init_Rtoa(ds)

    try:
        if hasattr(params, 'bands_l1') and params.bands_l1 != 'None':
            assert len(params.bands_l1) == len(ds.bands)
            ds = ds.assign_coords(
                bands=params.bands_l1,
            )
            if srf is not None:
                srf = rename(srf, params.bands_l1)
    except KeyError:
        pass

    # Central wavelength
    if 'cwav' not in ds:
        assert srf is not None
        ds['cwav'] = xr.DataArray(
            list(integrate_srf(srf, lambda x: x).values()),
            dims=['bands'],
            ).astype('float32')
    if ds.cwav.dtype == 'float64':
        ds['cwav'] = ds.cwav.astype('float32')
    
    # initialize bands_corr, bands_oc, bands_rw if they are defined from a callable
    if not isinstance(params.bands_corr, list):
        assert not isinstance(params.bands_oc, list)
        assert not isinstance(params.bands_rw, list)
        bands_level1 = ds.bands.values
        bands_rw = params.bands_rw(bands_level1)
        setattr(params, "bands_rw", bands_rw)
        setattr(params, "bands_corr", params.bands_corr(bands_rw))
        setattr(params, "bands_oc", params.bands_oc(bands_rw))

        # make sure there are no duplicates in bands_rw
        assert len(bands_rw) == len(set(bands_rw))
    
    # Store the params in the object attributes
    ds.attrs.update(params.items())

    return ds, srf


def compat(ds: xr.Dataset) -> xr.Dataset:
    '''
    Compatibility of new eoread inputs with current implementation.
    This compatibility function may be removed eventually.

    '''
    if 'bands_group' in ds:
        ds = ds.drop_vars('bands_group')

    for varname in ds:
        if ds[varname].dtype == 'float64':
            ds[varname] = ds[varname].astype('float32')
    
    if ds.platform.startswith('Sentinel-3') and ds.sensor == 'OLCI':
        # Consider moving global "Central wavelength" to attributes
        if 'wav' in ds:
            if 'cwav' in ds:
                ds = ds.drop_vars('cwav')
            ds = ds.rename(wav = 'cwav')

    assert 'wav' not in ds
        
    return ds


def run_polymer_dataset(
    ds: xr.Dataset,
    *,
    outputs: Literal["created_modified", "all", "tags", "named"] = "created_modified",
    outputs_tags: Optional[list[str]] = None,
    outputs_names: Optional[list[str]] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Polymer: main function at dataset level

    Arguments:
        ds: Input Level-1 dataset.
        outputs ({"created_modified", "all", "tags", "named"}):
            How to select output variables for the CompoundProcessor.
            - "created_modified": variables created or modified by the processors
            - "all": all variables (created, modified, and input)
            - "tags": variables with tags matching outputs_tags
            - "named": variables with names matching outputs_names
        outputs_tags: list of tags to filter output variables when outputs="tags".
            Available tags:
            - "level2": latitude, longitude, flags, rho_w, Rnir, Rgli, SPM
            - "ancillary": total_column_ozone, altitude, horizontal_wind, sea_level_pressure
            - "geometry": raa, sza, vza
            - "debug": cwav, Rtoa, rho_gc, rho_r, rho_rg, t_d, rho_rc,
                       logchl, logfb, niter, Ratm, Rwmod, eps,
                       total_column_ozone, altitude, horizontal_wind, sea_level_pressure
        outputs_names: list of variable names to include when outputs="named".
        **kwargs: additional arguments passed to Params.

    Returns:
        The processed dataset with Level-2 products.
    """
    ds = compat(ds)

    sensor = getattr(ds, 'sensor', None)
    platform = getattr(ds, 'platform', None)
    params = Params(sensor, platform=platform, **kwargs)

    if "srf_getter" in params.asdict():
        srf = get_SRF(ds, **params.asdict(), rename_method="bands")
    else:
        # empty dictionary when srfs are not provided
        srf = None

    ds, srf = init(ds, srf, params)

    # Bands selection (in ds and srf)
    ds = ds.sel(bands=params.bands_read()).chunk(bands=-1)
    if srf is not None:
        srf = srf[params.bands_read()]

    #
    # Build list of processors
    #
    processors = []

    # Geometry variable initialization
    processors.append(InitGeometry(ds, calc_air_mass=True, calc_scat_angle=True))
    
    # Flags initialization
    processors.append(
        FlagsInit(
            flags={
                GenericFlags.LAND: 1 << 0,
                GenericFlags.L1_INVALID: 1 << 2,
            },
            dtype="uint16",
            flag_reader=ds.attrs["_flag_reader"],
            flag_reader_kwargs=ds.attrs.get("_flag_reader_kwargs", {}),
            strict=False,
        )
    )
    
    # Set ancillary data: altitude
    if params.dem is not None:
        processors.append(DEM(ds, source=params.dem))
    else:
        assert "altitude" in ds

    # Andillary data: meteo/ozone
    if params.ancillary is not None:
        processors.append(ApplyAncillary(ds, params.ancillary))
    
    # Vicarious calibration
    processors.append(ApplyCalib(ds, 'Rtoa', params.calib))
    
    # Uncertainties initialization
    processors.append(InitUncertainties(ds, params))
    
    # Land mask
    if 'gsw_agg' in kwargs:
        processors.append(GSWLandMask(l1=ds, agg=kwargs['gsw_agg']))
    
    # Gaseous correction
    processors.append(
        Gaseous_correction(ds, srf, input_var="Rtoa", **dict(params.items()))
    )
    
    # Rayleigh correction
    processors.append(RayleighCorrection(srf=srf))
    
    # Rename RayleighCorrection output to Polymer naming conventions
    processors.append(RenameRayleigh(params.band_cloudmask))
    
    # Cloud mask
    processors.append(
        Cloud_mask(
            cm_input_var="Rprime",
            cm_band_nir=params.band_cloudmask,
            cm_flag_value=L2FLAGS["CLOUD_BASE"],
            cm_flag_name="CLOUD_BASE",
        )
    )
    
    # Sun glint calculation
    processors.append(CalcSunGlint())

    # Polymer solver — pass class + kwargs so each worker process can
    # reconstruct its own fresh Cython instances.
    processors.append(
        PolymerSolver(
            watermodel_cls=ParkRuddick,
            watermodel_kwargs={
                "directory": params.dir_common,
                "bbopt": params.bbopt,
                "min_abs": params.min_abs,
                "absorption": params.absorption,
            },
            params=params,
        )
    )

    # Tag Level-2 output variables for selection
    processors.append(TagOutputs(params))

    # Apply all processors to the input dataset (lazily)
    compound = CompoundProcessor(
        processors,
        outputs=outputs,
        outputs_tags=outputs_tags,
        outputs_names=outputs_names,
    )
    res = compound.map_blocks(ds)

    return res


def normalize_water_reflectance(
    ds: xr.Dataset,
    *,
    wind_speed: float = 5.0,
    sza0: float = 0.0,
    vza0: float = 0.0,
    raa0: float = 0.0,
    **kwargs,
) -> xr.Dataset:
    """
    Normalize water-leaving reflectance spectra using PolymerSolver.

    This function wraps PolymerSolver with atm_model='none', meaning no atmospheric
    correction is performed. The input water reflectance spectrum should be provided
    as "Rprime" in the input dataset.
    The solver will fit the water model parameters and return the normalized rho_w.

    Arguments:
        ds: xarray Dataset with at least these variables:
            - Rprime (y, x, bands): input water reflectance
            - wav (bands): wavelengths in nm
            - sza (y, x): solar zenith angle in degrees
            - vza (y, x): viewer zenith angle in degrees
            - raa (y, x): relative azimuth angle in degrees
            - horizontal_wind (y, x): wind speed in m/s (optional, defaults to wind_speed)
        wind_speed: default wind speed in m/s if horizontal_wind is not in ds
        sza0: solar zenith angle for geometry normalization (degrees, default 0.0)
        vza0: viewer zenith angle for geometry normalization (degrees, default 0.0)
        raa0: relative azimuth angle for geometry normalization (degrees, default 0.0)

    Returns:
        The input dataset augmented with rho_w, logchl, logfb and updated flags.
    """
    bands = ds.bands.values.tolist()

    params = Params(
        sensor='default',
        atm_model='none',
        bands_corr=bands,
        bands_oc=bands,
        bands_rw=bands,
        sza0=sza0,
        vza0=vza0,
        raa0=raa0,
        **kwargs,
    )
    params.finalize()

    # Create dummy variables needed by PolymerSolver
    if 'cwav' not in ds:
        ds = ds.assign(cwav=ds.wav)
    if 'Rprime_noglint' not in ds:
        ds = ds.assign(Rprime_noglint=ds.Rprime)
    if 'rho_r' not in ds:
        ds = ds.assign(rho_r=xr.zeros_like(ds.Rprime))
    if 'Tmol' not in ds:
        ds = ds.assign(Tmol=xr.ones_like(ds.Rprime))
    if 'Rgli' not in ds:
        ds = ds.assign(Rgli=xr.zeros_like(ds.sza))
    if 'Rnir' not in ds:
        ds = ds.assign(Rnir=ds.Rprime.isel(bands=-1))
    if 'horizontal_wind' not in ds:
        ds = ds.assign(horizontal_wind=xr.full_like(ds.sza, float(wind_speed)))
    if 'mus' not in ds:
        ds = ds.assign(mus=np.cos(np.radians(ds.sza)).astype('float32'))
    if 'muv' not in ds:
        ds = ds.assign(muv=np.cos(np.radians(ds.vza)).astype('float32'))
    if 'flags' not in ds:
        ds = ds.assign(flags=xr.zeros_like(ds.sza, dtype='uint16'))

    solver = PolymerSolver(
        watermodel_cls=ParkRuddick,
        watermodel_kwargs={'directory': params.dir_common},
        params=params,
    )
    solver.process_block(ds)

    return ds


class RenameRayleigh(BlockProcessor):
    """
    Rename RayleighCorrection output variables to Polymer naming conventions.
    
    Maps:
        - rho_rc (Rayleigh + glint corrected) -> Rprime
        - rho_gc - rho_r (Rayleigh only corrected) -> Rprime_noglint
        - t_d (total transmittance) -> Tmol
        - Rprime_noglint at NIR band -> Rnir
    """
    def __init__(self, band_cloudmask: int):
        self.band_cloudmask = band_cloudmask

    def input_vars(self) -> list[Var]:
        return [
            Var('rho_rc'),
            Var('rho_gc'),
            Var('rho_r'),
            Var('t_d'),
        ]

    def created_vars(self) -> list[Var]:
        return [
            Var('Rprime', dtype='float32', dims_like='rho_rc'),
            Var('Rprime_noglint', dtype='float32', dims_like='rho_gc'),
            Var('Tmol', dtype='float32', dims_like='t_d'),
            Var('Rnir', dtype='float32', dims=('y', 'x')),
        ]

    def process_block(self, block: xr.Dataset) -> None:
        block['Rprime'] = block['rho_rc'].astype('float32')
        block['Rprime_noglint'] = (block['rho_gc'] - block['rho_r']).astype('float32')
        block['Tmol'] = block['t_d'].astype('float32')
        block['Rnir'] = block['Rprime_noglint'].sel(bands=self.band_cloudmask).astype('float32')


class TagOutputs(BlockProcessor):
    """
    Tag Level-2 output variables for selection.

    Declares a fixed set of variables with tags=["level2"] so they can be
    selected using outputs="tags" and outputs_tags=["level2"] in CompoundProcessor.

    Tagged variables are: latitude, longitude, flags, rho_w.

    Variables are declared via modified_vars, which merges with created_vars
    from other processors through CompoundProcessor's merge logic.
    """
    def __init__(self, params):
        self.tags = {
            "latitude": ["level2"],
            "longitude": ["level2"],
            "flags": ["level2"],
            "rho_w": ["level2"],
            "total_column_ozone": ["ancillary", "debug"],
            "altitude": ["ancillary", "debug"],
            "horizontal_wind": ["ancillary", "debug"],
            "sea_level_pressure": ["ancillary", "debug"],
            "raa": ["geometry"],
            "sza": ["geometry"],
            "vza": ["geometry"],
            "cwav": ["level2", "debug"],
            "Rtoa": ["debug"],
            "rho_gc": ["debug"],
            "rho_r": ["debug"],
            "rho_rg": ["debug"],
            "t_d": ["debug"],
            "rho_rc": ["debug"],
            "Rnir": ["level2"],
            "Rgli": ["level2"],
            "logchl": ["debug"],
            "logfb": ["debug"],
            "SPM": ["level2"],
            "niter": ["debug"],
            "Ratm": ["debug"],
            "Rwmod": ["debug"],
            "eps": ["debug"],
        }
        if params.uncertainties:
            self.tags["rho_w_unc"] = ["level2"]
            self.tags["Rtoa_var"] = ["debug"]

    def modified_vars(self) -> list[Var]:
        return [Var(varname, tags=self.tags[varname]) for varname in self.tags]

    def process_block(self, block: xr.Dataset) -> None:
        pass


class ApplyCalib(BlockProcessor):
    """
    Apply calibration coefficients to a variable (in place).
    """
    def __init__(self, ds: xr.Dataset, varname: str, calib: dict|None):
        self.varname = varname
        self.calib = calib
        self.activate = calib is not None
        if self.activate and calib is not None:
            self.coeff = xr.DataArray(
                [calib[x] for x in ds.bands.data],
                dims=['bands'],
                coords={'bands': ds.bands.data},
            ).astype('float32')

    def modified_vars(self) -> list[Var]:
        if self.activate:
            return [Var(self.varname)]
        return []

    def process_block(self, block: xr.Dataset) -> None:
        coeff = self.coeff.sel(bands=block[self.varname].bands)
        block[self.varname] = block[self.varname] * coeff

