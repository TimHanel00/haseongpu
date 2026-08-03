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
    CrossSectionTable,
    FresnelInterface,
    InitialState,
    Material,
    MonteCarloASESolver,
    MonteCarloPumpSolver,
    PerfectTransmission,
    PumpSolver,
    RungeKutta4,
    Simulation,
    UnstructuredMesh,
    units,
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
    return UnstructuredMesh.fromTetrahedra(
        points,
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        volumeDomains=[10, 20],
        volumeDomainNames={10: "core", 20: "cap"},
        coordinateUnit=units.m,
    )


def materials():
    crossSections = CrossSectionTable.monochromatic(
        wavelength=1030 * units.nm,
        absorption=1.0e-20 * units.cm**2,
        emission=2.0e-20 * units.cm**2,
    )
    material = Material("Yb:YAG").addState(
        temperature=300 * units.K,
        refractiveIndex=1.82,
        fluorescenceLifetime=941 * units.us,
        crossSections=crossSections,
        metadata={"source": "synthetic test material"},
    )
    return (
        material.at(temperature=300 * units.K, activeIonDensity=2.76e26 / units.m**3, name="core"),
        material.at(temperature=300 * units.K, activeIonDensity=0.0 / units.m**3, name="cap"),
    )


def simulation(mesh):
    return Simulation(
        mesh=mesh,
        aseSolver=MonteCarloASESolver(),
        pumpSolver=MonteCarloPumpSolver(rayCount=16),
        timeIntegrator=RungeKutta4(),
        timeStepSize=1.0 * units.us,
        initialState=InitialState(0.25 * units.one),
    )


def complete_problem(interface_model=None):
    if interface_model is None:
        interface_model = PerfectTransmission()
    mesh = two_tet_mesh()
    core, cap = materials()
    configured = simulation(mesh)
    configured.addMaterial(core, domains=mesh.volume("core"))
    configured.addMaterial(cap, domains=mesh.volume("cap"))
    configured.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)
    configured.addInterface(interface_model, between=(core, cap))
    return configured


def test_multi_material_problem_compiles_to_dense_tables():
    compiled = complete_problem().resolveProblem()

    np.testing.assert_array_equal(compiled.cellMaterialId, [0, 1])
    assert [material.displayName for material in compiled.materials] == ["core", "cap"]
    assert np.count_nonzero(compiled.faceInterfaceId == 0) == 2
    np.testing.assert_array_equal(compiled.initialExcitationFraction, [0.25, 0.25])


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
    compiled = complete_problem(model).resolveProblem()

    assert compiled.interfaces[0].kind == model.kind


def test_backend_rejects_unwired_multi_material_features_before_launch(monkeypatch):
    configured = complete_problem(FresnelInterface())

    def unexpected_launch(*args, **kwargs):
        pytest.fail("unsupported problem reached the native transport")

    monkeypatch.setattr("pyInclude.openpmd.transport.run_simulation", unexpected_launch)
    with pytest.raises(NotImplementedError, match="multiple materials.*internal interface models: fresnel"):
        configured.step()


def test_future_capability_set_accepts_compiled_multi_material_interface():
    compiled = complete_problem(FresnelInterface()).resolveProblem()
    future = BackendCapabilities(
        multipleMaterials=True,
        internalInterfaceModels=frozenset({"fresnel"}),
    )

    assert compiled.unsupportedFeatures(future) == ()
    assert compiled.requireBackendSupport(future) is compiled


def test_solver_roles_accept_future_descriptors_but_current_adapter_rejects_them():
    class DeterministicASESolver(ASESolver):
        pass

    class DeterministicPumpSolver(PumpSolver):
        pass

    mesh = two_tet_mesh()
    core, _cap = materials()
    configured = Simulation(
        mesh=mesh,
        aseSolver=DeterministicASESolver(),
        pumpSolver=DeterministicPumpSolver(),
        timeIntegrator=RungeKutta4(),
        timeStepSize=1.0 * units.us,
        initialState=InitialState(0.25 * units.one),
    )
    configured.addMaterial(core, domains=mesh.volume("core", "cap"))
    configured.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)

    configured.resolveProblem()
    with pytest.raises(NotImplementedError, match="only Monte Carlo solvers"):
        configured.validateBackend()


def test_monte_carlo_ase_yaml_uses_explicit_snake_case_overrides(tmp_path):
    config = tmp_path / "phi-ase.yaml"
    config.write_text(
        """
experiment:
  min_rays: 1000
  max_rays: 10000
compute:
  backend: FromYaml
  ranks_per_node: 2
""",
        encoding="utf-8",
    )

    solver = MonteCarloASESolver.fromYaml(
        config,
        minRays=2500,
        backend="ExplicitOverride",
        ranksPerNode=4,
    )

    assert solver.minRays == 2500
    assert solver.maxRays == 10000
    assert solver.backend == "ExplicitOverride"
    assert solver.ranksPerNode == 4


def test_monte_carlo_ase_yaml_rejects_unknown_overrides(tmp_path):
    config = tmp_path / "phi-ase.yaml"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(TypeError, match="unknown.*made_up"):
        MonteCarloASESolver.fromYaml(config, made_up=1)


@pytest.mark.parametrize(
    "configure, match, error_type",
    [
        (
            lambda sim, core, cap: (
                sim.addMaterial(core, domains=sim.mesh.volume("core", "cap")),
                sim.addMaterial(cap, domains=sim.mesh.volume("cap")),
            ),
            "overlaps",
            ValueError,
        ),
        (
            lambda sim, core, cap: sim.addMaterial(core, domains=sim.mesh.volume("core")),
            "uncovered",
            ValueError,
        ),
        (
            lambda sim, core, cap: (
                sim.addMaterial(core, domains=sim.mesh.volume("core")),
                sim.addMaterial(cap, domains=sim.mesh.volume("cap")),
            ),
            "missing material interface",
            ValueError,
        ),
        (
            lambda sim, core, cap: sim.addMaterial(core, domains=sim.mesh.volume("missing")),
            "missing",
            KeyError,
        ),
    ],
)
def test_material_layout_validation(configure, match, error_type):
    mesh = two_tet_mesh()
    core, cap = materials()
    configured = simulation(mesh)

    with pytest.raises(error_type, match=match):
        configure(configured, core, cap)
        configured.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)
        configured.resolveProblem()


def test_same_material_condition_reuses_dense_material_id():
    mesh = two_tet_mesh()
    core, _cap = materials()
    configured = simulation(mesh)
    configured.addMaterial(core, domains=mesh.volume("core"))
    configured.addMaterial(core, domains=mesh.volume("cap"))
    configured.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)

    compiled = configured.resolveProblem()
    assert len(compiled.materials) == 1
    np.testing.assert_array_equal(compiled.cellMaterialId, [0, 0])


def test_three_material_chain_keeps_distinct_reciprocal_interface_ids():
    mesh = UnstructuredMesh.fromTetrahedra(
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
        cellConnectivity=[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
        volumeDomains=[10, 20, 30],
        coordinateUnit=units.m,
    )
    active, _passive = materials()
    base = Material("chain").addState(
        temperature=300 * units.K,
        refractiveIndex=active.refractiveIndex,
        fluorescenceLifetime=active.fluorescenceLifetime,
        crossSections=active.crossSections,
        metadata={"source": "synthetic chain"},
    )
    middle = base.at(temperature=300 * units.K, name="middle")
    right = base.at(temperature=300 * units.K, name="right")
    configured = simulation(mesh)
    configured.addMaterial(active, domains=mesh.volume(10))
    configured.addMaterial(middle, domains=mesh.volume(20))
    configured.addMaterial(right, domains=mesh.volume(30))
    configured.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)
    configured.addInterface(PerfectTransmission(), between=(middle, right))
    configured.addInterface(PerfectTransmission(), between=(active, middle))

    compiled = configured.resolveProblem()
    np.testing.assert_array_equal(compiled.cellMaterialId, [0, 1, 2])
    assert np.count_nonzero(compiled.faceInterfaceId == 0) == 2
    assert np.count_nonzero(compiled.faceInterfaceId == 1) == 2
    for cell, neighbor, expected_id in ((0, 1, 1), (1, 0, 1), (1, 2, 0), (2, 1, 0)):
        local_face = int(np.flatnonzero(mesh.neighborCells[cell] == neighbor)[0])
        assert compiled.faceInterfaceId[cell, local_face] == expected_id
