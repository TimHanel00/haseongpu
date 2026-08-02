# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace

import numpy as np
import pytest

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
    Pump,
    PumpSpectrum,
    RungeKutta4,
    Simulation,
    SurfacePumpInjector,
    UnstructuredMesh,
    writeParaviewState,
)


TETRAHEDRON = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def one_cell_simulation(**overrides):
    mesh = UnstructuredMesh.from_tetrahedra(
        TETRAHEDRON,
        [[0, 1, 2, 3]],
        surface_domains=[[1, 1, 1, 1]],
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
    arguments = {
        "mesh": mesh,
        "ase_solver": MonteCarloASESolver(backend="Host_Cpu_CpuSerial"),
        "pump_solver": MonteCarloPumpSolver(ray_count=16),
        "time_integrator": RungeKutta4(),
        "time_step_size": 1.0e-6,
        "initial_state": InitialState(0.0),
    }
    arguments.update(overrides)
    simulation = Simulation(**arguments)
    simulation.add_material(material, MaterialLayout("all"))
    simulation.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior"))
    return simulation


def test_mesh_topology_and_derived_arrays_are_immutable():
    mesh = UnstructuredMesh.from_tetrahedra(TETRAHEDRON, [[0, 1, 2, 3]])

    for values in (
        mesh.points,
        mesh.cell_connectivity,
        mesh.volume_domain_ids,
        mesh.surface_domain_ids,
        mesh.neighbor_cells,
        mesh.cellCenters,
    ):
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = 99


@pytest.mark.parametrize(
    "points,cells,match",
    [
        (TETRAHEDRON, [[0, 0, 1, 2]], "four distinct"),
        (TETRAHEDRON, [[0, 1, 2, 4]], "out-of-range"),
        (TETRAHEDRON * [1.0, 1.0, 0.0], [[0, 1, 2, 3]], "non-zero volume"),
        (TETRAHEDRON + [np.nan, 0.0, 0.0], [[0, 1, 2, 3]], "finite"),
        (
            np.vstack((TETRAHEDRON[:3], [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]])),
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]],
            "non-manifold",
        ),
    ],
)
def test_mesh_rejects_invalid_tetrahedra(points, cells, match):
    with pytest.raises((TypeError, ValueError), match=match):
        UnstructuredMesh.from_tetrahedra(points, cells)


@pytest.mark.parametrize("field,value", [("time_step_size", np.nan), ("time_step_size", np.inf), ("max_time", np.nan), ("max_time", np.inf)])
def test_simulation_rejects_nonfinite_time_controls(field, value):
    with pytest.raises(ValueError, match="finite and positive"):
        one_cell_simulation(**{field: value})


def test_missing_pump_error_does_not_freeze_configuration():
    simulation = one_cell_simulation()

    with pytest.raises(ValueError, match="requires at least one pump"):
        simulation.step()

    assert simulation.add_pump(
        Pump(total_power=1.0, spectrum=PumpSpectrum.monochromatic(940e-9)),
        SurfacePumpInjector(1),
    ) is simulation


def test_paraview_export_accepts_public_read_only_tet4_state(tmp_path):
    mesh = UnstructuredMesh.from_tetrahedra(TETRAHEDRON, [[0, 1, 2, 3]])
    state = SimpleNamespace(
        step=1,
        time=1.0e-6,
        topology=mesh,
        betaCells=np.asarray([0.1]),
        betaVolume=np.asarray([0.1]),
        phiAse=np.asarray([4.0]),
        dndtAse=np.asarray([-1.0]),
        dndtPump=np.asarray([2.0]),
    )

    handle = writeParaviewState(state, tmp_path)

    assert handle == tmp_path / "laserPumpCladding.pmd"
    assert handle.read_text(encoding="utf-8") == "laserPumpCladding_%06T.bp\n"
    assert (tmp_path / "laserPumpCladding_000001.bp").exists()
