#!/usr/bin/env python3
# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import numpy as np

import HASEonGPU
from HASEonGPU import (
    AbsorbingSurface,
    BoundaryLayout,
    CrossSectionTable,
    ExteriorBoundary,
    InitialState,
    MaterialDefinition,
    MaterialInstance,
    MaterialLayout,
    MonteCarloASESolver,
    MonteCarloPumpSolver,
    RungeKutta4,
    Simulation,
    UnstructuredMesh,
)


module_path = Path(HASEonGPU.__file__).resolve()
print("module:", module_path)
assert "site-packages" in str(module_path) or "dist-packages" in str(module_path), module_path

mesh = UnstructuredMesh.from_tetrahedra(
    np.eye(4, 3),
    [[0, 1, 2, 3]],
)
cross_sections = CrossSectionTable.monochromatic(
    wavelength=1030e-9,
    absorption=1.0e-24,
    emission=2.0e-24,
)
material = MaterialInstance(
    MaterialDefinition("gain", 1.8, 1.0e-3, cross_sections),
    active_ion_density=2.5e26,
)
simulation = Simulation(
    mesh=mesh,
    ase_solver=MonteCarloASESolver(backend="Host_Cpu_CpuSerial"),
    pump_solver=MonteCarloPumpSolver(ray_count=16),
    time_integrator=RungeKutta4(),
    time_step_size=1.0e-6,
    initial_state=InitialState(0.0),
)
simulation.add_material(material, MaterialLayout("all"))
simulation.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior"))
compiled = simulation.compile()

assert mesh.number_of_cells == 1
assert compiled.materials == (material,)
for legacy_name in (
    "CrossSectionData",
    "DomainMap",
    "GainMedium",
    "GainMediumGeometry",
    "Gmsh",
    "Grid",
    "LaserProperties",
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
