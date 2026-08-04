#!/usr/bin/env python3
# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import numpy as np

import HASEonGPU
from HASEonGPU import (
    AbsorbingSurface,
    CrossSectionTable,
    InitialState,
    Material,
    MonteCarloASESolver,
    MonteCarloPumpSolver,
    RungeKutta4,
    Simulation,
    UnstructuredMesh,
    units,
)


module_path = Path(HASEonGPU.__file__).resolve()
print("module:", module_path)
assert "site-packages" in str(module_path) or "dist-packages" in str(module_path), module_path

mesh = UnstructuredMesh.fromTetrahedra(
    np.eye(4, 3),
    [[0, 1, 2, 3]],
    coordinateUnit=units.m,
)
crossSections = CrossSectionTable.monochromatic(
    wavelength=1030 * units.nm,
    absorption=1.0e-20 * units.cm**2,
    emission=2.0e-20 * units.cm**2,
)
material = Material("gain").addState(
    temperature=300 * units.K,
    refractiveIndex=1.8,
    fluorescenceLifetime=1.0 * units.ms,
    crossSections=crossSections,
    metadata={"source": "installed frontend smoke"},
).at(
    temperature=300 * units.K,
    activeIonDensity=2.5e26 / units.m**3,
)
aseSolver = MonteCarloASESolver()
simulation = Simulation(
    mesh=mesh,
    aseSolver=aseSolver,
    pumpSolver=MonteCarloPumpSolver(rayCount=16),
    timeIntegrator=RungeKutta4(),
    timeStepSize=1.0 * units.us,
    initialState=InitialState(0.0 * units.one),
)
simulation.addMaterial(material, domains=mesh.volume(1))
simulation.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)
compiled = simulation.resolveProblem()

assert mesh.numberOfCells == 1
assert aseSolver.minRays == 100_000
assert aseSolver.maxRays == 100_000
assert compiled.materials == (material,)
np.testing.assert_array_equal(compiled.cellMaterialId, [0])
np.testing.assert_array_equal(compiled.faceBoundaryId, [[0, 0, 0, 0]])
np.testing.assert_array_equal(compiled.initialExcitationFraction, [0.0])
for legacy_name in (
    "CrossSectionData",
    "DomainMap",
    "GainMedium",
    "GainMediumGeometry",
    "Gmsh",
    "Grid",
    "LaserProperties",
    "MaterialDefinition",
    "MaterialInstance",
    "MeshTopology",
    "PhiASE",
    "SpectralDecomposition",
    "SurfaceOptics",
    "SurfaceDomainMap",
    "TimeSteppedSimulation",
    "TransportResult",
    "VolumeTopology",
    "calcGainFromState",
    "vtkWedge",
    "writeGainMediumVtk",
):
    assert not hasattr(HASEonGPU, legacy_name), legacy_name
