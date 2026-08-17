# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public Python convenience exports for HASEonGPU's openPMD frontend."""

__version__ = "2.2.0"

from ._runtime import activate_openpmd_python_provider as _activate_openpmd_python_provider

_activate_openpmd_python_provider()
del _activate_openpmd_python_provider

from .alpakaUtils import AlpakaBackends
from .openpmd import (
    BaseGroup,
    BaseSchema,
    GroupFieldSpec,
    OpenPmdBackends,
    PointSchema,
    PrimitiveFieldSpec,
    PrismSchema,
    TriangleSchema,
    backendFlat,
    unitDimension,
)
from .geometry import (
    DomainMap,
    Gmsh,
    Grid,
    MeshTopology,
    SurfaceDomainMap,
    SurfaceOptics,
    VolumeTopology,
)
from .physical import Domain, GainMedium, OpticalComponent
from material_library import CrossSectionTable, Material, MaterialLibrary
from hase_units import Quantity, Unit, units
from .laser import (
    CrossSectionData,
    GaussianPump,
    LaserProperties,
    PlanarPumpRelay,
    Pump,
    PumpAngularDistribution,
    PumpSpectrum,
    SpectralDecomposition,
    SuperGaussianPumpProfile,
    SurfacePumpInjector,
    UniformPumpProfile,
    integrate_pump_profile,
)
from .simulation import (
    PhiASE,
    Simulation,
    TimeStepState,
    autonomous_final,
)
from .structures import Result as TransportResult
from .gainMap import calcGainFromState
from .openpmd.paraview import writeParaviewState
from .vtkWedge import vtkWedge
from .timeIntegration import (
    ExplicitEuler,
    ExponentialEuler,
    FrozenPhiAseRungeKutta4,
    Heun,
    ImplicitEuler,
    Midpoint,
    RungeKutta4,
    TimeIntegrationSolver,
)
