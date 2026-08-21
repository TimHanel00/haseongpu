import numpy as np
import pytest

from hase_units import units
from material_library import CrossSectionTable, Material
from pyInclude.geometry import SurfaceOptics, VolumeTopology
from pyInclude.physical import (
    Domain,
    GainMedium,
    OpticalComponent,
    SURFACE,
    VOLUME,
    validateComponentOverlap,
)
from pyInclude.frontendState import projectFrontendState
from pyInclude.simulation import PhiASE, Simulation
from pyInclude.timeIntegration import ExplicitEuler


def topology(offset=0.0):
    points = np.asarray(
        [
            [offset + 0.0, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 0.0, 1.0, 0.0],
            [offset + 0.0, 0.0, 1.0],
            [offset + 1.0, 1.0, 1.0],
        ]
    )
    return VolumeTopology.fromTetrahedra(
        points,
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        cellDomains=[1, 2],
        metadata={"cellDomainNames": {1: "left", 2: "right"}},
    )


def singleTetrahedron(offset=0.0):
    return VolumeTopology.fromTetrahedra(
        np.asarray(
            [
                [offset + 0.0, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset + 0.0, 1.0, 0.0],
                [offset + 0.0, 0.0, 1.0],
            ]
        ),
        [[0, 1, 2, 3]],
    )
class HexRegionTopology:
    """Minimal six-face topology used to exercise the generic domain layer."""

    numberOfCells = 2
    neighborCells = np.asarray(
        [
            [-1, -1, -1, -1, -1, 1],
            [-1, -1, -1, -1, -1, 0],
        ],
        dtype=np.int32,
    )


def material(name="Yb:YAG"):
    return Material(
        materialName=name,
        temperature=293.15 * units.K,
        refractiveIndex=1.83,
        fluorescenceLifetime=0.941 * units.ms,
        crossSections=CrossSectionTable.monochromatic(
            wavelength=1030 * units.nm,
            absorption=1.0e-21 * units.cm**2,
            emission=2.0e-20 * units.cm**2,
        ),
        active=True,
        activeIonDensity=2.776e20 / units.cm**3,
    )


def test_domain_arithmetic_preserves_kind_and_spans_topologies():
    first = topology()
    second = topology(10.0)
    left = Domain.fromGmsh(first, "left")
    right = Domain.fromGmsh(first, "right")
    remote = Domain.fromTopology(second)

    combined = left + right + remote

    assert combined.entityKind == VOLUME
    np.testing.assert_array_equal(combined.maskFor(first), [True, True])
    np.testing.assert_array_equal(combined.maskFor(second), [True, True])
    np.testing.assert_array_equal((combined - right).maskFor(first), [True, False])


def test_domain_rejects_arithmetic_between_volume_and_surface():
    mesh = topology()
    with pytest.raises(TypeError, match="matching entity kinds"):
        _ = Domain.fromTopology(mesh) + Domain.fromTopology(mesh, entityKind=SURFACE)


def test_domain_is_structurally_immutable():
    mesh = topology()
    domain = Domain.fromGmsh(mesh, "left")

    with pytest.raises(AttributeError, match="immutable"):
        domain.entityKind = SURFACE
    with pytest.raises(AttributeError, match="immutable"):
        domain._shards = ()
    with pytest.raises(ValueError, match="read-only"):
        domain.maskFor(mesh)[0] = False


def test_domain_where_selects_typed_extreme_surface():
    mesh = topology()
    bottom = Domain.where(mesh, "z_min")
    top = Domain.where(mesh, "z_max")

    assert bottom.entityKind == SURFACE
    assert np.any(bottom.maskFor(mesh))
    assert np.any(top.maskFor(mesh))
    assert not np.any(bottom.maskFor(mesh) & top.maskFor(mesh))


def test_domain_boundary_removes_faces_inside_a_volume_union():
    mesh = topology()
    left = Domain.fromGmsh(mesh, "left")
    complete = left + Domain.fromGmsh(mesh, "right")

    leftBoundary = left.boundary().maskFor(mesh)
    completeBoundary = complete.boundary().maskFor(mesh)

    assert np.any(leftBoundary & (mesh.neighborCells >= 0))
    assert not np.any(completeBoundary & (mesh.neighborCells >= 0))
    assert np.count_nonzero(completeBoundary) == 6


def test_components_may_span_regions_or_select_a_shared_mesh():
    mesh = topology()
    left = OpticalComponent(material=material(), domain=Domain.fromGmsh(mesh, "left"), name="left")
    right = OpticalComponent(material=material(), domain=Domain.fromGmsh(mesh, "right"), name="right")
    independent = OpticalComponent(
        material=material(),
        domain=Domain.fromTopology(topology(10.0)),
        name="independent",
    )

    medium = GainMedium([left, right, independent])

    assert all(component.opticalRole == "gainElement" for component in medium.components)
    assert len(medium.domain.topologies) == 2
    np.testing.assert_array_equal(left.exteriorCells.maskFor(mesh), [True, False])


def test_one_component_may_span_independent_domain_bindings():
    combined = Domain([Domain.fromTopology(topology()), Domain.fromTopology(topology(10.0))])
    component = OpticalComponent(material=material(), domain=combined)

    assert component.domain is combined
    assert len(component.domain.topologies) == 2


def test_domain_frontend_accepts_non_tet_regions_but_state_projection_rejects_them():
    topology = HexRegionTopology()
    complete = Domain.fromTopology(topology)
    component = OpticalComponent(domain=complete, material=material())

    assert complete.maskFor(topology).tolist() == [True, True]
    assert np.count_nonzero(complete.boundary().maskFor(topology)) == 10
    with pytest.raises(NotImplementedError, match="Tet4 VolumeTopology"):
        projectFrontendState(GainMedium([component]))


def test_component_surface_optics_must_target_its_boundary():
    mesh = topology()
    with pytest.raises(TypeError):
        OpticalComponent(material=material(), topology=mesh)

    component = OpticalComponent(material=material(), domain=Domain.fromTopology(mesh))
    exterior = Domain.fromTopology(mesh, entityKind=SURFACE)

    assert component.assignSurfaceOptics(exterior, SurfaceOptics(reflectivity=0.5)) is component
    assert len(component.surfaceOptics) == 1
    with pytest.raises(ValueError, match="must not overlap"):
        component.assignSurfaceOptics(exterior, SurfaceOptics(reflectivity=0.25))


def test_components_on_different_topologies_may_touch_but_not_overlap():
    leftTopology = singleTetrahedron()
    touchingTopology = singleTetrahedron(1.0)
    overlappingTopology = singleTetrahedron(1.0 - 1.0e-6)
    left = OpticalComponent(material=material(), domain=Domain.fromTopology(leftTopology))
    touching = OpticalComponent(
        material=material(), domain=Domain.fromTopology(touchingTopology)
    )
    overlapping = OpticalComponent(
        material=material(), domain=Domain.fromTopology(overlappingTopology)
    )

    validateComponentOverlap((left, touching))
    with pytest.raises(ValueError, match="positive volume"):
        validateComponentOverlap((left, overlapping))


def test_resolved_material_remains_mutable_and_revalidatable():
    resolved = material()
    resolved.refractiveIndex = 1.9
    assert resolved.validate() is resolved
    assert resolved.refractiveIndex == pytest.approx(1.9)

    resolved.activeIonDensity = -1.0 / units.cm**3
    with pytest.raises(ValueError, match="non-negative"):
        resolved.validate()


def test_material_activity_is_authoritative_for_population_dynamics():
    active = material()
    active.activeIonDensity = 0.0 / units.cm**3
    assert active.validate().isActive

    component = OpticalComponent(
        material=active,
        domain=Domain.fromTopology(topology()),
    )
    with pytest.raises(ValueError, match="positive activeIonDensity"):
        projectFrontendState(GainMedium([component]))

    with pytest.raises(ValueError, match="passive materials require zero"):
        Material(
            materialName="invalid passive material",
            temperature=293.15 * units.K,
            refractiveIndex=1.45,
            fluorescenceLifetime=None,
            crossSections=None,
            active=False,
            activeIonDensity=1.0 / units.cm**3,
        )


def test_frontend_state_projection_preserves_shared_adjacency_and_concatenates_independent_meshes():
    shared = topology()
    shared_material = material()
    left = OpticalComponent(material=shared_material, domain=Domain.fromGmsh(shared, "left"))
    right = OpticalComponent(material=shared_material, domain=Domain.fromGmsh(shared, "right"))
    remote = OpticalComponent(
        material=shared_material,
        domain=Domain.fromTopology(topology(10.0)),
    )

    state = projectFrontendState(GainMedium([left, right, remote]), initialExcitation=0.25)

    assert state.topology.numberOfCells == 4
    assert state.topology.numberOfPoints == 10
    assert state.topology.neighborCells[0, 2] == 1
    assert np.all(state.get("betaVolume").value == 0.25)
    assert shared_material.activeIonDensity.toValue(units.cm**-3) == pytest.approx(2.776e20)
    assert shared_material.crossSections.wavelengths.size == 1


def test_frontend_state_projection_preserves_full_component_point_layout_including_unused_points():
    source = topology()
    mesh = VolumeTopology.fromTetrahedra(
        np.vstack((source.points, [[99.0, 99.0, 99.0]])),
        source.cellPointIndices,
    )
    resolved = material()

    state = projectFrontendState(
        GainMedium(
            [
                OpticalComponent(
                    material=resolved,
                    domain=Domain.fromTopology(mesh),
                )
            ]
        )
    )

    assert state.topology.numberOfPoints == mesh.numberOfPoints
    np.testing.assert_array_equal(state.topology.cellPointIndices, mesh.cellPointIndices)


def test_frontend_state_projection_rejects_overlaps_and_accepts_distinct_materials():
    mesh = topology()
    selected = Domain.fromGmsh(mesh, "left")
    first = OpticalComponent(material=material(), domain=selected)
    overlap = OpticalComponent(material=first.material, domain=selected)
    with pytest.raises(ValueError, match="must not overlap"):
        projectFrontendState(GainMedium([first, overlap]))

    secondMaterial = material("second material")
    secondMaterial.refractiveIndex = 1.7
    second = OpticalComponent(material=secondMaterial, domain=Domain.fromGmsh(mesh, "right"))
    state = projectFrontendState(GainMedium([first, second]))
    assert state.topology.numberOfCells == 2
    assert first.material is not second.material
    assert first.material.refractiveIndex == pytest.approx(1.83)
    assert second.material.refractiveIndex == pytest.approx(1.7)
    assert first.material.crossSections is not second.material.crossSections


def test_passive_component_material_stays_on_physical_graph():
    mesh = topology()
    active = material()
    gainComponent = OpticalComponent(
        material=active,
        domain=Domain.fromGmsh(mesh, "right"),
    )
    passive = Material(
        materialName="cladding",
        temperature=293.15 * units.K,
        refractiveIndex=1.45,
        fluorescenceLifetime=None,
        crossSections=None,
        active=False,
        bulkAttenuation=5.5 / units.cm,
    )
    claddingComponent = OpticalComponent(
        material=passive,
        domain=Domain.fromGmsh(mesh, "left"),
    )
    medium = GainMedium([gainComponent])

    state = projectFrontendState(
        medium,
        initialExcitation=0.25,
        opticalComponents=[gainComponent, claddingComponent],
    )

    np.testing.assert_array_equal(state.get("betaVolume").value, [0.0, 0.25])
    assert claddingComponent.material is passive
    assert not claddingComponent.material.isActive
    assert passive.bulkAttenuation.toValue(units.cm**-1) == pytest.approx(5.5)
    passive.bulkAttenuation = 4.0 / units.cm
    assert passive.bulkAttenuation.toValue(units.cm**-1) == pytest.approx(4.0)


def test_lossless_passive_state_projection():
    mesh = topology()
    gainComponent = OpticalComponent(
        material=material(), domain=Domain.fromGmsh(mesh, "right")
    )
    unspecified = Material(
        materialName="unspecified passive material",
        temperature=293.15 * units.K,
        refractiveIndex=1.45,
        fluorescenceLifetime=None,
        crossSections=None,
        active=False,
    )
    passiveComponent = OpticalComponent(
        material=unspecified, domain=Domain.fromGmsh(mesh, "left")
    )

    state = projectFrontendState(
        GainMedium([gainComponent]),
        opticalComponents=[gainComponent, passiveComponent],
    )

    assert unspecified.isTransparent
    np.testing.assert_array_equal(state.get("betaVolume").value, [0.0, 0.0])
    assert passiveComponent.material.bulkAttenuation is None


def test_frontend_state_projection_accepts_heterogeneous_passive_attenuation():
    gainComponent = OpticalComponent(
        material=material(),
        domain=Domain.fromTopology(topology()),
    )
    passiveComponents = [
        OpticalComponent(
            material=Material(
                materialName=f"passive material {attenuation}",
                temperature=293.15 * units.K,
                refractiveIndex=1.45,
                fluorescenceLifetime=None,
                crossSections=None,
                active=False,
                bulkAttenuation=attenuation / units.cm,
            ),
            domain=Domain.fromTopology(topology(offset=2.0 * index)),
        )
        for index, attenuation in enumerate((1.0, 2.0), start=1)
    ]

    state = projectFrontendState(
        GainMedium([gainComponent]),
        opticalComponents=[gainComponent, *passiveComponents],
    )
    assert state.topology.numberOfCells == 6
    assert [
        component.material.bulkAttenuation.toValue(units.cm**-1)
        for component in passiveComponents
    ] == pytest.approx([1.0, 2.0])


def test_simulation_keeps_non_gain_components_and_requires_exact_excitation_coverage():
    gainMesh = topology()
    gainMaterial = material()
    gainComponent = OpticalComponent(
        material=gainMaterial,
        domain=Domain.fromTopology(gainMesh),
    )
    windowMaterial = Material(
        materialName="window",
        temperature=293.15 * units.K,
        refractiveIndex=1.5,
        fluorescenceLifetime=None,
        crossSections=None,
        active=False,
    )
    windowMesh = topology(10.0)
    window = OpticalComponent(
        material=windowMaterial,
        domain=Domain.fromTopology(windowMesh),
    )
    medium = GainMedium([gainComponent])

    simulation = Simulation(
        opticalComponents=[gainComponent, window],
        gainMedium=medium,
        initialExcitation=0.1,
        phiASE=PhiASE(ase_steps=0),
        timeIntegrator=ExplicitEuler(),
        timeStepSize=1.0e-6,
        simulationSteps=1,
    )
    assert simulation.opticalComponents == (gainComponent, window)
    assert window.opticalRole is None
    assert simulation.opticalComponents[1].material is windowMaterial
    assert simulation.opticalComponents[1].material.refractiveIndex == pytest.approx(1.5)
    assert simulation.opticalComponents[1].material.isTransparent
    np.testing.assert_array_equal(
        simulation._simulationState.get("betaVolume").value,
        [0.1, 0.1, 0.0, 0.0],
    )
    np.testing.assert_array_equal(simulation.cellMask(gainComponent.domain), [True, True, False, False])
    np.testing.assert_array_equal(simulation.cellMask(window.domain), [False, False, True, True])
    assert simulation.exteriorSurface.entityKind == SURFACE
    assert len(simulation.exteriorSurface.topologies) == 2
    assert not np.any(
        simulation.exteriorSurface.maskFor(gainMesh)
        & (gainMesh.neighborCells >= 0)
    )
    assert not np.any(
        simulation.exteriorSurface.maskFor(windowMesh)
        & (windowMesh.neighborCells >= 0)
    )

    exterior = (gainComponent.domain + window.domain).boundary()
    simulationWithExterior = Simulation(
        opticalComponents=[gainComponent, window],
        gainMedium=medium,
        exteriorSurface=exterior,
        initialExcitation=0.1,
        phiASE=PhiASE(ase_steps=0),
        timeIntegrator=ExplicitEuler(),
        timeStepSize=1.0e-6,
        simulationSteps=1,
    )
    assert simulationWithExterior.exteriorSurface is exterior

    with pytest.raises(ValueError, match="non-empty surface Domain"):
        Simulation(
            opticalComponents=[gainComponent, window],
            gainMedium=medium,
            exteriorSurface=gainComponent.domain,
            initialExcitation=0.1,
            phiASE=PhiASE(ase_steps=0),
            timeIntegrator=ExplicitEuler(),
            timeStepSize=1.0e-6,
            simulationSteps=1,
        )

    with pytest.raises(ValueError, match="cover the GainMedium exactly once"):
        Simulation(
            opticalComponents=[gainComponent, window],
            gainMedium=medium,
            initialExcitation={Domain.fromGmsh(gainMesh, "left"): 0.1},
            phiASE=PhiASE(ase_steps=0),
            timeIntegrator=ExplicitEuler(),
            timeStepSize=1.0e-6,
            simulationSteps=1,
        )
