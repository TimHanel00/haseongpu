# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Load named Tet4 domains from gmsh and attach one optical material."""

from pathlib import Path
from tempfile import TemporaryDirectory

import gmsh

try:
    from ._source_tree_import import ensure_hase_importable
except ImportError:
    from _source_tree_import import ensure_hase_importable

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
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
        wavelength=1030 * units.nm,
        absorption=1.2e-21 * units.cm**2,
        emission=2.48e-20 * units.cm**2,
    )
    yag = Material("Yb:YAG").addState(
        temperature=300 * units.K,
        refractiveIndex=1.82,
        fluorescenceLifetime=941 * units.us,
        crossSections=cross_sections,
        metadata={"source": "synthetic gmsh example"},
    )
    crystal = yag.at(
        temperature=300 * units.K,
        activeIonDensity=2.76e20 / units.cm**3,
    )
    simulation = Simulation(
        mesh=mesh,
        aseSolver=MonteCarloASESolver(backend="Host_Cpu_CpuSerial"),
        pumpSolver=MonteCarloPumpSolver(rayCount=100_000),
        timeIntegrator=RungeKutta4(),
        timeStepSize=10 * units.us,
        initialState=InitialState(0 * units.one),
    )
    simulation.addMaterial(crystal, domains=mesh.volume("crystal"))
    simulation.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)
    return simulation


def main():
    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "minimal_crystal.msh"
        write_minimal_gmsh_mesh(filename)
        mesh = UnstructuredMesh.fromFile(filename, coordinateUnit=units.cm)
        simulation = build_simulation(mesh)
        problem = simulation.resolveProblem()
        print(f"gmsh topology: {mesh.numberOfCells} tetrahedra")
        print(f"volume domains: {mesh.volumeDomainNames}")
        print(f"surface domains: {mesh.surfaceDomainNames}")
        print(f"material table: {[material.displayName for material in problem.materials]}")
        print(f"backend-compatible: {problem.unsupportedFeatures() == ()}")


if __name__ == "__main__":
    main()
