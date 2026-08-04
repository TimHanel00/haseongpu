# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace

import numpy as np
import pytest

from HASEonGPU import (
    AbsorbingSurface,
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
    autonomousFinal,
    writeParaviewState,
    units,
)


TETRAHEDRON = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def one_cell_simulation(**overrides):
    mesh = UnstructuredMesh.fromTetrahedra(
        TETRAHEDRON,
        [[0, 1, 2, 3]],
        surfaceDomains=[[1, 1, 1, 1]],
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
        metadata={"source": "synthetic mesh test"},
    ).at(
        temperature=300 * units.K,
        activeIonDensity=2.5e26 / units.m**3,
    )
    arguments = {
        "mesh": mesh,
        "aseSolver": MonteCarloASESolver(),
        "pumpSolver": MonteCarloPumpSolver(rayCount=16),
        "timeIntegrator": RungeKutta4(),
        "timeStepSize": 1.0 * units.us,
        "initialState": InitialState(0.0 * units.one),
    }
    arguments.update(overrides)
    simulation = Simulation(**arguments)
    simulation.addMaterial(material, domains=mesh.volume(1))
    simulation.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)
    return simulation


def test_mesh_topology_and_derived_arrays_are_immutable():
    mesh = UnstructuredMesh.fromTetrahedra(
        TETRAHEDRON, [[0, 1, 2, 3]], coordinateUnit=units.m
    )

    for values in (
        mesh.points,
        mesh.cellConnectivity,
        mesh.volumeDomainIds,
        mesh.surfaceDomainIds,
        mesh.neighborCells,
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
        UnstructuredMesh.fromTetrahedra(points, cells, coordinateUnit=units.m)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeStepSize", np.nan * units.us),
        ("timeStepSize", np.inf * units.us),
        ("maxTime", np.nan * units.us),
        ("maxTime", np.inf * units.us),
    ],
)
def test_simulation_rejects_nonfinite_time_controls(field, value):
    with pytest.raises(ValueError, match="finite and positive"):
        one_cell_simulation(**{field: value})


def test_missing_pump_error_does_not_freeze_configuration():
    simulation = one_cell_simulation()

    with pytest.raises(ValueError, match="requires at least one pump"):
        simulation.step()

    assert simulation.addPump(
        Pump(totalPower=1.0 * units.W, spectrum=PumpSpectrum.monochromatic(940 * units.nm)),
        SurfacePumpInjector(simulation.mesh.surface(1)),
    ) is simulation


def _add_test_pump(simulation):
    return simulation.addPump(
        Pump(totalPower=1.0 * units.W, spectrum=PumpSpectrum.monochromatic(940 * units.nm)),
        SurfacePumpInjector(simulation.mesh.surface(1)),
    )


def _raw_cell_state(step, cell_count=1):
    values = np.full(cell_count, 0.1 * step, dtype=np.float64)
    return SimpleNamespace(
        step=step,
        time=step * 1.0e-6,
        betaVolume=values,
        phiAse=values + 1.0,
        standardError=values * 0.01,
        relativeStandardError=values * 0.02,
        totalRays=np.full(cell_count, step, dtype=np.uint32),
        dndtAse=-values,
        dndtPump=values,
        aseResult=object(),
    )


def test_autonomous_final_is_a_thin_output_schedule_helper():
    assert autonomousFinal(5) == (5,)
    with pytest.raises(ValueError, match="positive"):
        autonomousFinal(0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"executionMode": "invalid"}, "executionMode"),
        ({"outputSteps": (2, 1)}, "strictly increasing"),
        ({"outputFields": ("point_beta",)}, "unsupported outputFields"),
        ({"controlFields": ("beta_volume",)}, "synchronized-debug"),
        (
            {"executionMode": "synchronized-debug", "outputSteps": (1,)},
            "emits every completed step",
        ),
    ],
)
def test_simulation_rejects_unsupported_execution_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        one_cell_simulation(**overrides)


def test_autonomous_run_materializes_only_selected_cell_snapshots(monkeypatch):
    simulation = one_cell_simulation(
        outputSteps=(2, 4),
        outputFields=("beta_volume", "phi_ase"),
    )
    _add_test_pump(simulation)
    observed = []
    simulation.onStep(observed.append)

    def fake_run(simulation, *, steps, on_state, **kwargs):
        assert steps == 4
        assert simulation.outputSteps == (2, 4)
        states = [_raw_cell_state(step) for step in simulation.outputSteps]
        for state in states:
            on_state(state)
        return states

    monkeypatch.setattr("pyInclude.openpmd.transport.run_simulation", fake_run)
    simulation.step(4)

    assert [state.step for state in observed] == [2, 4]
    assert observed[-1].excitationFraction.shape == (1,)
    assert not hasattr(observed[-1], "sampledExcitationFraction")
    assert simulation.currentStep == 4


def test_synchronized_debug_callback_returns_explicit_beta_control(monkeypatch):
    simulation = one_cell_simulation(
        executionMode="synchronized-debug",
        controlFields=("beta_volume",),
    )
    _add_test_pump(simulation)
    simulation.onControl(
        lambda state: {"beta_volume": np.minimum(state.excitationFraction, 0.15)}
    )
    controls = []

    def fake_run(simulation, *, steps, on_state, **kwargs):
        states = [_raw_cell_state(step) for step in range(1, steps + 1)]
        for state in states:
            controls.append(on_state(state))
        return states

    monkeypatch.setattr("pyInclude.openpmd.transport.run_simulation", fake_run)
    simulation.step(3)

    np.testing.assert_allclose(controls[0]["beta_volume"], [0.1])
    np.testing.assert_allclose(controls[1]["beta_volume"], [0.15])


def test_paraview_export_accepts_public_read_only_tet4_state(tmp_path):
    mesh = UnstructuredMesh.fromTetrahedra(
        TETRAHEDRON, [[0, 1, 2, 3]], coordinateUnit=units.m
    )
    state = SimpleNamespace(
        step=1,
        time=1.0 * units.us,
        mesh=mesh,
        excitationFraction=np.asarray([0.1]),
        phiAse=np.asarray([4.0]),
        dExcitationDtAse=np.asarray([-1.0]),
        dExcitationDtPump=np.asarray([2.0]),
    )

    handle = writeParaviewState(state, tmp_path)

    assert handle == tmp_path / "laserPumpCladding.pmd"
    assert handle.read_text(encoding="utf-8") == "laserPumpCladding_%06T.bp\n"
    assert (tmp_path / "laserPumpCladding_000001.bp").exists()
