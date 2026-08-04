# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public Python API for HASEonGPU."""

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
    unitDimension,
)
from .mesh import MeshSelection, UnstructuredMesh
from material_library import (
    CrossSectionTable,
    LegacyMaterialTextWarning,
    Material,
    MaterialCondition,
    MaterialLibrary,
    MaterialState,
    TemperatureInterpolationWarning,
    loadBuiltinMaterials,
)
#: Preloaded built-in :class:`MaterialLibrary`; select entries by mapping key,
#: for example ``materialLibrary["YbYAG"]``.
materialLibrary = loadBuiltinMaterials()
from hase_units import Quantity, Unit, units
from .solvers import ASESolver, PumpSolver
from .problem import (
    AbsorbingSurface,
    BackendCapabilities,
    ResolvedProblem,
    ConstantReflectivitySurface,
    currentBackendCapabilities,
    ExteriorBoundaryModel,
    FresnelInterface,
    InitialState,
    MaterialInterfaceModel,
    PerfectTransmission,
)
from .laser import (
    GaussianPump,
    MonteCarloPumpSolver,
    PlanarPumpRelay,
    Pump,
    PumpAngularDistribution,
    PumpSpectrum,
    SuperGaussianPumpProfile,
    SurfacePumpInjector,
    UniformPumpProfile,
    integratePumpProfile,
)
from .simulation import MonteCarloASESolver, Simulation, TimeStepState, autonomousFinal
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

from .openpmd.paraview import writeParaviewState
from .vtk_state import writeVtkState


__all__ = [
    "ASESolver",
    "AbsorbingSurface",
    "AlpakaBackends",
    "BackendCapabilities",
    "BaseGroup",
    "BaseSchema",
    "currentBackendCapabilities",
    "ResolvedProblem",
    "ConstantReflectivitySurface",
    "CrossSectionTable",
    "ExplicitEuler",
    "ExponentialEuler",
    "ExteriorBoundaryModel",
    "FresnelInterface",
    "FrozenPhiAseRungeKutta4",
    "GaussianPump",
    "GroupFieldSpec",
    "Heun",
    "ImplicitEuler",
    "InitialState",
    "LegacyMaterialTextWarning",
    "Material",
    "MaterialCondition",
    "MaterialLibrary",
    "MaterialState",
    "MaterialInterfaceModel",
    "MeshSelection",
    "Midpoint",
    "MonteCarloASESolver",
    "MonteCarloPumpSolver",
    "OpenPmdBackends",
    "PerfectTransmission",
    "PlanarPumpRelay",
    "PointSchema",
    "PrimitiveFieldSpec",
    "PrismSchema",
    "Pump",
    "PumpAngularDistribution",
    "PumpSolver",
    "PumpSpectrum",
    "Quantity",
    "RungeKutta4",
    "Simulation",
    "SuperGaussianPumpProfile",
    "SurfacePumpInjector",
    "TimeIntegrationSolver",
    "TimeStepState",
    "TemperatureInterpolationWarning",
    "TriangleSchema",
    "UniformPumpProfile",
    "UnstructuredMesh",
    "Unit",
    "autonomousFinal",
    "integratePumpProfile",
    "materialLibrary",
    "unitDimension",
    "writeParaviewState",
    "writeVtkState",
    "units",
]
