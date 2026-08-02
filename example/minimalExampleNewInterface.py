# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal one-material simulation using the PICMI-aligned Python API."""

import numpy as np

from _source_tree_import import ensure_hase_importable

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
    BoundaryLayout,
    ConstantReflectivitySurface,
    CrossSectionTable,
    ExteriorBoundary,
    InitialState,
    MaterialDefinition,
    MaterialInstance,
    MaterialLayout,
    MonteCarloASESolver,
    MonteCarloPumpSolver,
    Pump,
    PumpSpectrum,
    RungeKutta4,
    Simulation,
    SurfacePumpInjector,
    UnstructuredMesh,
)


def print_state(state):
    print(
        f"step={state.step:03d} time={state.time:.3e}s "
        f"mean_excitation={state.excitation_fraction.mean():.6e}"
    )


def build_simulation():
    # docs:start: mesh
    # Mesh topology and domain identity contain no optical material data.
    mesh = UnstructuredMesh.from_tetrahedra(
        points=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        cell_connectivity=[[0, 1, 2, 3]],
        volume_domains=[10],
        surface_domains=np.ones((1, 4), dtype=np.int32),
        volume_domain_names={10: "crystal"},
        surface_domain_names={1: "exterior"},
    )
    # docs:end: mesh

    # docs:start: material
    cross_sections = CrossSectionTable(
        wavelengths=[900e-9, 1030e-9],
        absorption=[1.1e-25, 1.2e-25],
        emission=[2.0e-24, 2.48e-24],
    )
    yag = MaterialDefinition(
        name="Yb:YAG",
        refractive_index=1.82,
        fluorescence_lifetime=941e-6,
        cross_sections=cross_sections,
    )
    crystal = MaterialInstance(yag, active_ion_density=2.76e26)
    # docs:end: material

    # docs:start: simulation
    simulation = Simulation(
        mesh=mesh,
        ase_solver=MonteCarloASESolver(
            min_rays=100_000,
            max_rays=100_000,
            backend="Host_Cpu_CpuSerial",
        ),
        pump_solver=MonteCarloPumpSolver(ray_count=100_000),
        time_integrator=RungeKutta4(),
        time_step_size=1e-5,
        initial_state=InitialState(excitation_fraction=0.0),
        max_time=1e-3,
    )
    simulation.add_material(crystal, MaterialLayout("crystal"))
    simulation.add_boundary(
        ExteriorBoundary(ConstantReflectivitySurface(reflectivity=0.0)),
        BoundaryLayout("exterior"),
    )
    # docs:end: simulation

    # docs:start: pump
    simulation.add_pump(
        Pump(total_power=16e3, spectrum=PumpSpectrum.monochromatic(940e-9)),
        SurfacePumpInjector("exterior"),
    )
    # docs:end: pump
    simulation.on_step(print_state)
    return simulation


def main():
    simulation = build_simulation()

    # Compilation checks domain coverage without launching the native backend.
    problem = simulation.compile()
    print(
        f"compiled {problem.mesh.number_of_cells} cell(s) with "
        f"{len(problem.materials)} material(s)"
    )

    # step() validates native-backend support and then performs the run.
    simulation.step(3)


if __name__ == "__main__":
    main()
