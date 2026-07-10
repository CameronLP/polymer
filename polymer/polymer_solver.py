#!/usr/bin/env python
# encoding: utf-8

import xarray as xr
from core.tools import Var
from core.process.blockwise import BlockProcessor

from polymer.polymer_main import PolymerSolverCy


class PolymerSolver(BlockProcessor):
    """
    Polymer solver as a BlockProcessor.

    Wraps PolymerSolverCy (the Cython-optimized solver) so it can be used
    within the BlockProcessor / CompoundProcessor pipeline.

    To support `scheduler="processes"` all Cython objects (`watermodel`
    and `PolymerSolverCy`) are passed as class + constructor kwargs rather than
    as instances, so that each worker process reconstructs its own instance.

    Example:

        PolymerSolver(
            watermodel_cls=ParkRuddick,
            watermodel_kwargs={
                'directory': params.dir_common,
                'bbopt': params.bbopt,
                'min_abs': params.min_abs,
                'absorption': params.absorption,
            },
            params=params,
        )
    """

    def __init__(self,
                 watermodel_cls=None,
                 watermodel_kwargs=None,
                 params=None):
        """
        Arguments:
            watermodel_cls: water model class (e.g. ParkRuddick).
                Picklable reference — each worker reconstructs its own instance.
            watermodel_kwargs: dict of kwargs passed to ``watermodel_cls``.
            params: Params object with sensor-specific configuration.
        """
        self.watermodel_cls = watermodel_cls
        self.watermodel_kwargs = dict(watermodel_kwargs) if watermodel_kwargs else {}
        self._watermodel = None
        self._solver = None
        self.params = params

    # ------------------------------------------------------------------
    # Pickle support — drop Cython instances, keep class + kwargs
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_watermodel', None)
        state.pop('_solver', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Ensure Cython objects are None after unpickling
        # (they were stripped by __getstate__)
        self.__dict__.setdefault('_watermodel', None)
        self.__dict__.setdefault('_solver', None)

    # ------------------------------------------------------------------
    # Lazy properties — reconstruct Cython objects on first access
    # ------------------------------------------------------------------

    @property
    def watermodel(self):
        """Lazily construct the water model on first access (per worker)."""
        if self._watermodel is None and self.watermodel_cls is not None:
            self._watermodel = self.watermodel_cls(**self.watermodel_kwargs)
        return self._watermodel

    @property
    def solver(self):
        """Lazily construct PolymerSolverCy on first access (per worker)."""
        if self._solver is None:
            self._solver = PolymerSolverCy(self.watermodel, self.params)
        return self._solver

    # ------------------------------------------------------------------
    # BlockProcessor interface
    # ------------------------------------------------------------------

    def input_vars(self) -> list[Var]:
        """Variables consumed by the solver."""
        return [
            Var('Rprime'),
            Var('Rprime_noglint'),
            Var('rho_r'),
            Var('Rnir'),
            Var('Rgli'),
            Var('Tmol'),
            Var('cwav'),
            Var('sza'),
            Var('vza'),
            Var('raa'),
            Var('mus'),
            Var('muv'),
            Var('horizontal_wind'),
            Var('flags'),
        ]

    def created_vars(self) -> list[Var]:
        """New variables produced by the solver."""
        variables = [
            Var('logchl', dtype='float32', dims=('y', 'x')),
            Var('fa', dtype='float32', dims=('y', 'x')),
            Var('logfb', dtype='float32', dims=('y', 'x')),
            Var('SPM', dtype='float32', dims=('y', 'x')),
            Var('niter', dtype='uint32', dims=('y', 'x')),
            Var('rho_w', dtype='float32', dims=('y', 'x', 'bands')),
            Var('Ratm', dtype='float32', dims=('y', 'x', 'bands')),
            Var('Rwmod', dtype='float32', dims=('y', 'x', 'bands')),
            Var('eps', dtype='float32', dims=('y', 'x')),
        ]
        if self.params.uncertainties:
            variables.extend([
                Var('logchl_unc', dtype='float32', dims=('y', 'x')),
                Var('logfb_unc', dtype='float32', dims=('y', 'x')),
                Var('rho_w_unc', dtype='float32', dims=('y', 'x', 'bands')),
            ])
        return variables

    def modified_vars(self) -> list[Var]:
        """Existing variables that are updated in place."""
        return [Var('flags')]

    def process_block(self, block: xr.Dataset) -> None:
        """
        Process a single block by delegating to PolymerSolverCy.apply.

        PolymerSolverCy.apply expects float32 inputs, so we ensure all
        float64 variables are cast before calling the Cython solver.
        The solver modifies the block in place.
        """
        # Ensure all float variables are float32 for Cython compatibility
        for varname in list(block.data_vars):
            if block[varname].dtype == 'float64':
                block[varname] = block[varname].astype('float32')

        self.solver.apply(block)
