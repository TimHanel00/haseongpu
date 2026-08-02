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
    backendFlat,
    unitDimension,
)
from .mesh import UnstructuredMesh
from .materials import CrossSectionTable, MaterialDefinition, MaterialInstance
from .solvers import ASESolver, PumpSolver
from .problem import (
    AbsorbingSurface,
    BackendCapabilities,
    BoundaryLayout,
    CompiledProblem,
    ConstantReflectivitySurface,
    CURRENT_BACKEND_CAPABILITIES,
    ExteriorBoundary,
    ExteriorBoundaryModel,
    FresnelInterface,
    InitialState,
    MaterialInterface,
    MaterialInterfaceModel,
    MaterialInterfaceLayout,
    MaterialLayout,
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
    integrate_pump_profile,
)
from .simulation import MonteCarloASESolver, Simulation, TimeStepState
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


__all__ = [
    "ASESolver",
    "AbsorbingSurface",
    "AlpakaBackends",
    "BackendCapabilities",
    "BaseGroup",
    "BaseSchema",
    "BoundaryLayout",
    "CURRENT_BACKEND_CAPABILITIES",
    "CompiledProblem",
    "ConstantReflectivitySurface",
    "CrossSectionTable",
    "ExplicitEuler",
    "ExponentialEuler",
    "ExteriorBoundary",
    "ExteriorBoundaryModel",
    "FresnelInterface",
    "FrozenPhiAseRungeKutta4",
    "GaussianPump",
    "GroupFieldSpec",
    "Heun",
    "ImplicitEuler",
    "InitialState",
    "MaterialDefinition",
    "MaterialInstance",
    "MaterialInterface",
    "MaterialInterfaceLayout",
    "MaterialInterfaceModel",
    "MaterialLayout",
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
    "RungeKutta4",
    "Simulation",
    "SuperGaussianPumpProfile",
    "SurfacePumpInjector",
    "TimeIntegrationSolver",
    "TimeStepState",
    "TriangleSchema",
    "UniformPumpProfile",
    "UnstructuredMesh",
    "backendFlat",
    "integrate_pump_profile",
    "unitDimension",
    "writeParaviewState",
]
