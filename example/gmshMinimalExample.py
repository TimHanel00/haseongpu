# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Construct a Tet4 component from named Gmsh physical groups."""

from pathlib import Path
from tempfile import TemporaryDirectory

import gmsh

from _source_tree_import import ensure_hase_importable

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
    CrossSectionTable,
    Domain,
    GainMedium,
    Material,
    OpticalComponent,
    PhiASE,
    Pump,
    PumpSpectrum,
    RungeKutta4,
    Simulation,
    SurfacePumpInjector,
    VolumeTopology,
    units,
)


def writeMinimalGmshMesh(filename):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("hase_gmsh_component")
        volume = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], 1)
        gmsh.model.setPhysicalName(3, 1, "Crystal")
        boundary = gmsh.model.getBoundary([(3, volume)], oriented=False)
        bottom = []
        top = []
        for dimension, tag in boundary:
            center = gmsh.model.occ.getCenterOfMass(dimension, tag)
            if abs(center[2]) < 1.0e-12:
                bottom.append(tag)
            elif abs(center[2] - 1.0) < 1.0e-12:
                top.append(tag)
        for tag, name, surfaces in ((2, "PumpInput", bottom), (3, "PumpOutput", top)):
            gmsh.model.addPhysicalGroup(2, surfaces, tag)
            gmsh.model.setPhysicalName(2, tag, name)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(filename))
    finally:
        gmsh.finalize()


def main():
    with TemporaryDirectory() as temporaryDirectory:
        meshPath = Path(temporaryDirectory) / "component.msh"
        writeMinimalGmshMesh(meshPath)
        topology = VolumeTopology.fromFile(meshPath, format="gmsh")
        crystalDomain = Domain.fromGmsh(topology, "Crystal", entityKind="volume")
        pumpInput = Domain.fromGmsh(topology, "PumpInput", entityKind="surface")

        material = Material(
            materialName="example gain material",
            temperature=293.15 * units.K,
            refractiveIndex=1.83,
            fluorescenceLifetime=0.941 * units.ms,
            crossSections=CrossSectionTable.monochromatic(
                wavelength=940 * units.nm,
                absorption=7.8e-21 * units.cm**2,
                emission=1.9e-21 * units.cm**2,
            ),
            active=True,
            activeIonDensity=2.776e20 / units.cm**3,
        )
        crystal = OpticalComponent(domain=crystalDomain, material=material)
        gainMedium = GainMedium([crystal])
        pump = Pump(
            total_power=100.0,
            spectrum=PumpSpectrum.monochromatic(940e-9),
            ray_count=10_000,
            pump_steps=3,
        )
        simulation = Simulation(
            opticalComponents=[crystal],
            gainMedium=gainMedium,
            phiASE=PhiASE(
                forwardRayCount=1000,
                repetitions=1,
                backend="Host_Cpu_CpuSerial",
                ase_steps=3,
            ),
            timeIntegrator=RungeKutta4(),
            timeStepSize=1e-5,
            simulationSteps=3,
        ).addPump(pump, SurfacePumpInjector(pumpInput))
        print(f"Gmsh cells: {topology.numberOfCells}")
        simulation.step()
        print(f"last completed step: {simulation.getLastState().step}")


if __name__ == "__main__":
    main()
