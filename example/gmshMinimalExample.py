# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Load named Tet4 domains from gmsh and attach one optical material."""

from pathlib import Path
from tempfile import TemporaryDirectory

import gmsh

from _source_tree_import import ensure_hase_importable

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
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


def write_minimal_gmsh_mesh(filename):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("hase_crystal")
        volume = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 0.25)
        gmsh.model.occ.synchronize()
        surfaces = [tag for dimension, tag in gmsh.model.getBoundary([(3, volume)]) if dimension == 2]
        gmsh.model.addPhysicalGroup(3, [volume], 10)
        gmsh.model.setPhysicalName(3, 10, "crystal")
        gmsh.model.addPhysicalGroup(2, surfaces, 20)
        gmsh.model.setPhysicalName(2, 20, "exterior")
        gmsh.model.mesh.generate(3)
        gmsh.write(str(filename))
    finally:
        gmsh.finalize()


def build_simulation(mesh):
    cross_sections = CrossSectionTable.monochromatic(
        wavelength=1030e-9,
        absorption=1.2e-25,
        emission=2.48e-24,
    )
    crystal = MaterialInstance(
        MaterialDefinition("Yb:YAG", 1.82, 941e-6, cross_sections),
        active_ion_density=2.76e26,
    )
    simulation = Simulation(
        mesh=mesh,
        ase_solver=MonteCarloASESolver(backend="Host_Cpu_CpuSerial"),
        pump_solver=MonteCarloPumpSolver(ray_count=100_000),
        time_integrator=RungeKutta4(),
        time_step_size=1e-5,
        initial_state=InitialState(0.0),
    )
    simulation.add_material(crystal, MaterialLayout("crystal"))
    simulation.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("exterior"))
    return simulation


def main():
    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "minimal_crystal.msh"
        write_minimal_gmsh_mesh(filename)
        mesh = UnstructuredMesh.from_file(filename)
        simulation = build_simulation(mesh)
        problem = simulation.compile()
        print(f"gmsh topology: {mesh.number_of_cells} tetrahedra")
        print(f"volume domains: {mesh.volume_domain_names}")
        print(f"surface domains: {mesh.surface_domain_names}")
        print(f"material table: {[material.display_name for material in problem.materials]}")
        print(f"backend-compatible: {problem.unsupported_features() == ()}")


if __name__ == "__main__":
    main()
