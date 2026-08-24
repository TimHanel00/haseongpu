from dataclasses import dataclass

import numpy as np
import pytest

from hase_units import units
from hase_transport import PrimitiveDescription, TransportComposer, field, reference
from material_library import CrossSectionTable, Material as PhysicalMaterial
from pyInclude.geometry import VolumeTopology
from pyInclude.openpmd.graph import _selectedFields
from pyInclude.physical import Domain, GainMedium, OpticalComponent
from pyInclude.simulation import PhiASE, Simulation
from pyInclude.timeIntegration import ExplicitEuler


@dataclass
class Material:
    refractiveIndex: float

    def _transportDescription(self):
        return PrimitiveDescription(
            "material",
            fields=(field("refractiveIndex"),),
        )


@dataclass
class Component:
    material: Material
    name: str

    def _transportDescription(self):
        return PrimitiveDescription(
            "opticalComponent",
            fields=(field("name"),),
            references=(reference("material"),),
        )


@dataclass
class Assembly:
    components: tuple[Component, ...]

    def _transportDescription(self):
        return PrimitiveDescription(
            "simulation",
            references=(reference("components", many=True),),
        )


def test_composer_deduplicates_shared_primitive_references():
    material = Material(1.82)
    graph = TransportComposer().compose(
        Assembly((Component(material, "gain"), Component(material, "window")))
    )

    assert graph.root == "simulation"
    assert [node.typeName for node in graph.nodes].count("material") == 1

    root = graph.node("simulation")
    first, second = (graph.node(path) for path in root.references["components"])
    assert first.references["material"] == second.references["material"]


def test_new_primitive_field_is_declared_only_in_its_description():
    material = Material(1.91)
    graph = TransportComposer().compose(material)

    field_spec, value = graph.node("material").fields["refractiveIndex"]
    assert field_spec.name == "refractiveIndex"
    assert value == 1.91


def test_transport_description_is_not_a_public_frontend_method():
    material = Material(1.5)

    assert not hasattr(material, "transportDescription")
    assert hasattr(material, "_transportDescription")


def test_physical_graph_serializes_shared_material_data_once():
    table = CrossSectionTable.monochromatic(
        wavelength=1030 * units.nm,
        absorption=1.0e-21 * units.cm**2,
        emission=2.0e-20 * units.cm**2,
    )
    material = PhysicalMaterial(
        materialName="shared gain material",
        temperature=293.15 * units.K,
        refractiveIndex=1.8,
        fluorescenceLifetime=1.0e-3 * units.s,
        crossSections=table,
        active=True,
        activeIonDensity=2.76e20 / units.cm**3,
    )

    def component(offset):
        topology = VolumeTopology.fromTetrahedra(
            np.asarray(
                [
                    [offset, 0.0, 0.0],
                    [offset + 1.0, 0.0, 0.0],
                    [offset, 1.0, 0.0],
                    [offset, 0.0, 1.0],
                ]
            ),
            [[0, 1, 2, 3]],
        )
        return OpticalComponent(material=material, domain=Domain.fromTopology(topology))

    components = (component(0.0), component(10.0))
    simulation = Simulation(
        opticalComponents=components,
        gainMedium=GainMedium(components),
        phiASE=PhiASE(backend="test-backend", ase_steps=0),
        timeIntegrator=ExplicitEuler(),
        timeStepSize=1.0e-6,
        simulationSteps=1,
    )
    graph = TransportComposer().compose(simulation)

    assert [node.typeName for node in graph.nodes].count("material") == 1
    assert [node.typeName for node in graph.nodes].count("crossSectionTable") == 1
    for topology in (node for node in graph.nodes if node.typeName == "volumeTopology"):
        point_spec, points = topology.fields["points"]
        _face_area_spec, face_areas = topology.fields["faceAreas"]
        _cell_volume_spec, cell_volumes = topology.fields["cellVolumes"]
        connectivity_spec, connectivity = topology.fields["cellPointIndices"]
        assert point_spec.axes == ("coordinate", "point")
        assert points.toValue(units.m).shape == (3, 4)
        assert face_areas.toValue(units.m**2).shape == (4, 1)
        assert cell_volumes.toValue(units.m**3).shape == (1,)
        assert connectivity_spec.axes == ("localVertex", "cell")
        assert np.asarray(connectivity).shape == (4, 1)
    phi_ase = next(node for node in graph.nodes if node.typeName == "phiAse")
    assert "crossSections" not in phi_ase.references

    dynamic_paths = {
        f"{node.path}/{name}"
        for node, name, _spec, _value in _selectedFields(graph, dynamicOnly=True)
    }
    assert dynamic_paths == {
        "simulation/currentStep",
        "simulation/currentTime",
        "objects/excitationState/0/values",
    }
    assert not any("volumeTopology" in path for path in dynamic_paths)

    excitationNode = next(node for node in graph.nodes if node.typeName == "excitationState")
    simulation._simulationState.get("betaVolume").value[:] = 0.75
    simulation._refreshExcitationState()
    refreshed = dict(
        (f"{node.path}/{name}", value)
        for node, name, _spec, value in _selectedFields(graph, dynamicOnly=True)
    )
    assert excitationNode.owner is simulation.excitationState
    np.testing.assert_allclose(refreshed[f"{excitationNode.path}/values"][0], 0.75)

    simulation.controlFields = ("cross_sections",)
    tableNode = next(node for node in graph.nodes if node.typeName == "crossSectionTable")
    controlledDynamicPaths = {
        f"{node.path}/{name}"
        for node, name, _spec, _value in _selectedFields(graph, dynamicOnly=True)
    }
    assert controlledDynamicPaths == dynamic_paths | {
        f"{tableNode.path}/wavelengths",
        f"{tableNode.path}/absorption",
        f"{tableNode.path}/emission",
    }
    table = tableNode.owner
    table.replaceSamples(
        [900.0, 950.0, 1000.0] * units.nm,
        [1.0e-21, 2.0e-21, 3.0e-21] * units.cm**2,
        [4.0e-21, 5.0e-21, 6.0e-21] * units.cm**2,
    )
    resized = dict(
        (f"{node.path}/{name}", value)
        for node, name, _spec, value in _selectedFields(graph, dynamicOnly=True)
    )
    assert tableNode.owner is table
    np.testing.assert_array_equal(
        resized[f"{tableNode.path}/wavelengths"].toValue(units.nm),
        [900.0, 950.0, 1000.0],
    )
    np.testing.assert_array_equal(
        resized[f"{tableNode.path}/absorption"].toValue(units.cm**2),
        [1.0e-21, 2.0e-21, 3.0e-21],
    )
    np.testing.assert_array_equal(
        resized[f"{tableNode.path}/emission"].toValue(units.cm**2),
        [4.0e-21, 5.0e-21, 6.0e-21],
    )

    previousWavelengths = table.wavelengths
    previousAbsorption = table.absorption
    previousEmission = table.emission
    with pytest.raises(ValueError, match="same length"):
        table.replaceSamples(
            [900.0, 950.0] * units.nm,
            [1.0e-21] * units.cm**2,
            [4.0e-21, 5.0e-21] * units.cm**2,
        )
    assert table.wavelengths is previousWavelengths
    assert table.absorption is previousAbsorption
    assert table.emission is previousEmission
