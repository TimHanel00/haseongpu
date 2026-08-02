# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from HASEonGPU import (
    AbsorbingSurface,
    ASESolver,
    BackendCapabilities,
    BoundaryLayout,
    CrossSectionTable,
    ExteriorBoundary,
    FresnelInterface,
    InitialState,
    MaterialDefinition,
    MaterialInstance,
    MaterialInterface,
    MaterialInterfaceLayout,
    MaterialLayout,
    MonteCarloASESolver,
    MonteCarloPumpSolver,
    PerfectTransmission,
    PumpSolver,
    RungeKutta4,
    Simulation,
    UnstructuredMesh,
)


def two_tet_mesh():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    return UnstructuredMesh.from_tetrahedra(
        points,
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        volume_domains=[10, 20],
        volume_domain_names={10: "core", 20: "cap"},
    )


def materials():
    cross_sections = CrossSectionTable.monochromatic(
        wavelength=1030e-9,
        absorption=1.0e-24,
        emission=2.0e-24,
    )
    definition = MaterialDefinition(
        name="Yb:YAG",
        refractive_index=1.82,
        fluorescence_lifetime=941e-6,
        cross_sections=cross_sections,
    )
    return (
        MaterialInstance(definition, active_ion_density=2.76e26, name="core"),
        MaterialInstance(definition, active_ion_density=0.0, name="cap"),
    )


def simulation(mesh):
    return Simulation(
        mesh=mesh,
        ase_solver=MonteCarloASESolver(backend="Host_Cpu_CpuSerial"),
        pump_solver=MonteCarloPumpSolver(ray_count=16),
        time_integrator=RungeKutta4(),
        time_step_size=1.0e-6,
        initial_state=InitialState(0.25),
    )


def complete_problem(interface_model=None):
    if interface_model is None:
        interface_model = PerfectTransmission()
    mesh = two_tet_mesh()
    core, cap = materials()
    configured = simulation(mesh)
    configured.add_material(core, MaterialLayout("core"))
    configured.add_material(cap, MaterialLayout("cap"))
    configured.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior"))
    configured.add_interface(
        MaterialInterface(interface_model),
        MaterialInterfaceLayout((core, cap)),
    )
    return configured


def test_multi_material_problem_compiles_to_dense_tables():
    compiled = complete_problem().compile()

    np.testing.assert_array_equal(compiled.cell_material_id, [0, 1])
    assert [material.display_name for material in compiled.materials] == ["core", "cap"]
    assert np.count_nonzero(compiled.face_interface_id == 0) == 2
    np.testing.assert_array_equal(compiled.initial_excitation_fraction, [0.25, 0.25])


def test_legacy_material_mesh_aggregate_is_not_public():
    import HASEonGPU

    for name in (
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
        assert not hasattr(HASEonGPU, name)


@pytest.mark.parametrize("model", [PerfectTransmission(), FresnelInterface()])
def test_internal_interface_models_survive_frontend_compilation(model):
    compiled = complete_problem(model).compile()

    assert compiled.interfaces[0].model.kind == model.kind


def test_backend_rejects_unwired_multi_material_features_before_launch(monkeypatch):
    configured = complete_problem(FresnelInterface())

    def unexpected_launch(*args, **kwargs):
        pytest.fail("unsupported problem reached the native transport")

    monkeypatch.setattr("pyInclude.openpmd.transport.runSimulation", unexpected_launch)
    with pytest.raises(NotImplementedError, match="multiple materials.*internal interface models: fresnel"):
        configured.step()


def test_future_capability_set_accepts_compiled_multi_material_interface():
    compiled = complete_problem(FresnelInterface()).compile()
    future = BackendCapabilities(
        multiple_materials=True,
        internal_interface_models=frozenset({"fresnel"}),
    )

    assert compiled.unsupported_features(future) == ()
    assert compiled.require_backend_support(future) is compiled


def test_solver_roles_accept_future_descriptors_but_current_adapter_rejects_them():
    class DeterministicASESolver(ASESolver):
        pass

    class DeterministicPumpSolver(PumpSolver):
        pass

    mesh = two_tet_mesh()
    core, _cap = materials()
    configured = Simulation(
        mesh=mesh,
        ase_solver=DeterministicASESolver(),
        pump_solver=DeterministicPumpSolver(),
        time_integrator=RungeKutta4(),
        time_step_size=1.0e-6,
        initial_state=InitialState(0.25),
    )
    configured.add_material(core, MaterialLayout("all"))
    configured.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior"))

    configured.compile()
    with pytest.raises(NotImplementedError, match="only Monte Carlo solvers"):
        configured.validate_backend()


def test_monte_carlo_ase_yaml_uses_explicit_snake_case_overrides(tmp_path):
    config = tmp_path / "phi-ase.yaml"
    config.write_text(
        """
experiment:
  minRays: 1000
  maxRays: 10000
compute:
  backend: FromYaml
  nPerNode: 2
""",
        encoding="utf-8",
    )

    solver = MonteCarloASESolver.from_yaml(
        config,
        min_rays=2500,
        backend="ExplicitOverride",
        ranks_per_node=4,
    )

    assert solver.minRays == 2500
    assert solver.maxRays == 10000
    assert solver.backend == "ExplicitOverride"
    assert solver.nPerNode == 4


def test_monte_carlo_ase_yaml_rejects_unknown_overrides(tmp_path):
    config = tmp_path / "phi-ase.yaml"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(TypeError, match="unexpected.*made_up"):
        MonteCarloASESolver.from_yaml(config, made_up=1)


@pytest.mark.parametrize(
    "configure, match, error_type",
    [
        (
            lambda sim, core, cap: (
                sim.add_material(core, MaterialLayout("all")),
                sim.add_material(cap, MaterialLayout("cap")),
            ),
            "overlaps",
            ValueError,
        ),
        (
            lambda sim, core, cap: sim.add_material(core, MaterialLayout("core")),
            "uncovered",
            ValueError,
        ),
        (
            lambda sim, core, cap: (
                sim.add_material(core, MaterialLayout("core")),
                sim.add_material(cap, MaterialLayout("cap")),
            ),
            "missing material interface",
            ValueError,
        ),
        (
            lambda sim, core, cap: sim.add_material(core, MaterialLayout("missing")),
            "missing",
            KeyError,
        ),
    ],
)
def test_material_layout_validation(configure, match, error_type):
    mesh = two_tet_mesh()
    core, cap = materials()
    configured = simulation(mesh)
    configure(configured, core, cap)
    configured.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior"))

    with pytest.raises(error_type, match=match):
        configured.compile()


def test_same_material_instance_reuses_dense_material_id():
    mesh = two_tet_mesh()
    core, _cap = materials()
    configured = simulation(mesh)
    configured.add_material(core, MaterialLayout("core"))
    configured.add_material(core, MaterialLayout("cap"))
    configured.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior"))

    compiled = configured.compile()
    assert len(compiled.materials) == 1
    np.testing.assert_array_equal(compiled.cell_material_id, [0, 0])


def test_three_material_chain_keeps_distinct_reciprocal_interface_ids():
    mesh = UnstructuredMesh.from_tetrahedra(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 0.0],
            ]
        ),
        cell_connectivity=[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
        volume_domains=[10, 20, 30],
    )
    active, _passive = materials()
    middle = MaterialInstance(active.definition, name="middle")
    right = MaterialInstance(active.definition, name="right")
    configured = simulation(mesh)
    configured.add_material(active, MaterialLayout(10))
    configured.add_material(middle, MaterialLayout(20))
    configured.add_material(right, MaterialLayout(30))
    configured.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior"))
    configured.add_interface(
        MaterialInterface(PerfectTransmission(), name="middle-right"),
        MaterialInterfaceLayout((middle, right)),
    )
    configured.add_interface(
        MaterialInterface(PerfectTransmission(), name="active-middle"),
        MaterialInterfaceLayout((active, middle)),
    )

    compiled = configured.compile()
    np.testing.assert_array_equal(compiled.cell_material_id, [0, 1, 2])
    assert np.count_nonzero(compiled.face_interface_id == 0) == 2
    assert np.count_nonzero(compiled.face_interface_id == 1) == 2
    for cell, neighbor, expected_id in ((0, 1, 1), (1, 0, 1), (1, 2, 0), (2, 1, 0)):
        local_face = int(np.flatnonzero(mesh.neighbor_cells[cell] == neighbor)[0])
        assert compiled.face_interface_id[cell, local_face] == expected_id


def test_single_material_adapter_converts_si_and_delegates(monkeypatch):
    from types import SimpleNamespace

    from HASEonGPU import Pump, PumpSpectrum, SurfacePumpInjector

    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh = UnstructuredMesh.from_tetrahedra(
        points,
        [[0, 1, 2, 3]],
        surface_domains=[[1, 2, 2, 2]],
        surface_domain_names={1: "pump_input", 2: "outer"},
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
    configured = simulation(mesh)
    configured.initial_state = InitialState(0.2)
    configured.add_material(material, MaterialLayout("all"))
    configured.add_boundary(ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("pump_input"))
    configured.add_boundary(
        ExteriorBoundary(
            model=__import__("HASEonGPU").ConstantReflectivitySurface(
                reflectivity=0.3,
                exterior_refractive_index=1.5,
            )
        ),
        BoundaryLayout("outer"),
    )
    configured.add_pump(
        Pump(total_power=1.0, spectrum=PumpSpectrum.monochromatic(1030e-9)),
        SurfacePumpInjector("pump_input"),
    )
    captured = {}

    def fake_run(legacy, **kwargs):
        captured["legacy"] = legacy
        return [
            SimpleNamespace(
                step=1,
                time=1.0e-6,
                betaCells=np.asarray([0.3]),
                betaVolume=np.asarray([0.3]),
                phiAse=np.asarray([4.0]),
                dndtAse=np.asarray([-1.0]),
                dndtPump=np.asarray([2.0]),
                aseResult=None,
            )
        ]

    monkeypatch.setattr("pyInclude.openpmd.transport.runSimulation", fake_run)
    configured.step()

    legacy = captured["legacy"]
    assert legacy.gainMedium.get("nTot").value == pytest.approx(2.5e20)
    assert legacy.gainMedium.get("crystalTFluo").value == pytest.approx(1.0e-3)
    np.testing.assert_allclose(legacy.crossSections.crossSectionAbsorption, [1.0e-20])
    np.testing.assert_array_equal(legacy.gainMedium.topology.faceBoundaries, [[1, 2, 2, 2]])
    np.testing.assert_allclose(legacy.gainMedium.get("surfaceReflectivity").value[[1, 2]], [0.0, 0.3])
    np.testing.assert_allclose(
        legacy.gainMedium.get("surfaceRefractiveIndexOutside").value[[1, 2]],
        [1.0, 1.5],
    )
    np.testing.assert_allclose(
        legacy.gainMedium.get("surfaceRefractiveIndexInside").value[[1, 2]],
        [1.0, 1.8],
    )
    assert legacy.pump.sources[0].surfaceDomains == ("pump_input",)
    np.testing.assert_allclose(legacy.gainMedium.get("betaVolume").value, [0.3])
    state = configured.get_last_state()
    assert state.topology is mesh
    np.testing.assert_allclose(state.excitation_fraction, [0.3])
