# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal one-material simulation using the PICMI-aligned Python API."""

import numpy as np

try:
    from ._source_tree_import import ensure_hase_importable
except ImportError:
    from _source_tree_import import ensure_hase_importable

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
    ConstantReflectivitySurface,
    CrossSectionTable,
    InitialState,
    Material,
    MonteCarloASESolver,
    MonteCarloPumpSolver,
    Pump,
    PumpSpectrum,
    RungeKutta4,
    Simulation,
    SurfacePumpInjector,
    UnstructuredMesh,
    units,
)


def print_state(state):
    print(
        f"step={state.step:03d} time={float(state.time.toValue(units.s)):.3e}s "
        f"mean_excitation={state.excitationFraction.mean():.6e}"
    )


def build_simulation():
    # docs:start: mesh
    # Mesh topology and domain identity contain no optical material data.
    mesh = UnstructuredMesh.fromTetrahedra(
        points=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        cellConnectivity=[[0, 1, 2, 3]],
        volumeDomains=[10],
        surfaceDomains=np.ones((1, 4), dtype=np.int32),
        volumeDomainNames={10: "crystal"},
        surfaceDomainNames={1: "exterior"},
        coordinateUnit=units.cm,
    )
    # docs:end: mesh

    # docs:start: material
    cross_sections = CrossSectionTable(
        wavelengths=np.asarray([900, 1030]) * units.nm,
        absorption=np.asarray([1.1e-21, 1.2e-21]) * units.cm**2,
        emission=np.asarray([2.0e-20, 2.48e-20]) * units.cm**2,
    )
    yag = Material("Yb:YAG").addState(
        temperature=300 * units.K,
        refractiveIndex=1.82,
        fluorescenceLifetime=941 * units.us,
        crossSections=cross_sections,
        metadata={"source": "synthetic minimal example"},
    )
    crystal = yag.at(
        temperature=300 * units.K,
        activeIonDensity=2.76e20 / units.cm**3,
    )
    # docs:end: material

    # docs:start: simulation
    simulation = Simulation(
        mesh=mesh,
        aseSolver=MonteCarloASESolver(
            minRays=100_000,
            maxRays=100_000,
            backend="Host_Cpu_CpuSerial",
        ),
        pumpSolver=MonteCarloPumpSolver(rayCount=100_000),
        timeIntegrator=RungeKutta4(),
        timeStepSize=10 * units.us,
        initialState=InitialState(excitationFraction=0 * units.one),
        maxTime=1 * units.ms,
    )
    simulation.addMaterial(crystal, domains=mesh.volume("crystal"))
    simulation.addBoundary(
        ConstantReflectivitySurface(reflectivity=0.0),
        domains=mesh.exteriorFaces,
    )
    # docs:end: simulation

    # docs:start: pump
    simulation.addPump(
        Pump(totalPower=16 * units.kW, spectrum=PumpSpectrum.monochromatic(940 * units.nm)),
        SurfacePumpInjector(mesh.exteriorFaces),
    )
    # docs:end: pump
    simulation.onStep(print_state)
    return simulation


def main():
    simulation = build_simulation()

    # Problem resolution checks domain coverage without launching the backend.
    problem = simulation.resolveProblem()
    print(
        f"resolved {problem.mesh.numberOfCells} cell(s) with "
        f"{len(problem.materials)} material(s)"
    )

    # step() validates native-backend support and then performs the run.
    simulation.step(3)


if __name__ == "__main__":
    main()
