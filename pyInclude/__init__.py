# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public Python convenience exports for HASEonGPU."""

__version__ = "2.1.0"


from .alpakaUtils import AlpakaBackends
from .openpmd import (
    BaseGroup,
    BaseSchema,
    GroupFieldSpec,
    PointSchema,
    PrimitiveFieldSpec,
    PrismSchema,
    TriangleSchema,
    backendFlat,
    unitDimension,
)
from .geometry import GainMedium, GainMediumGeometry, Gmsh, Grid, MeshTopology, VolumeTopology, writeGainMediumVtk
from .laser import CrossSectionData, LaserProperties, PumpProperties, SpectralDecomposition
from .simulation import (
    ConnectivityAverageBetaVolumeMapper,
    LegacyGridDataBetaVolumeMapper,
    PhiASE,
    Simulation,
    TimeStepState,
    TimeSteppedSimulation,
)
from .gainMap import calcGainFromState
from .structures import Result as TransportResult
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
