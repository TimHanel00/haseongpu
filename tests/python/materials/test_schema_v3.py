import yaml

from hase_units import units
from material_library import Material

from pyInclude import Domain, GainMedium, OpticalComponent, Simulation

from test_physical_api import material


def config():
    return {
        "schema_version": 3,
        "topologies": {
            "assembly": {
                "from_tetrahedra": {
                    "points": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    "cell_point_indices": [[0, 1, 2, 3], [1, 2, 3, 4]],
                    "cell_domains": [1, 2],
                    "metadata": {"cellDomainNames": {1: "left", 2: "right"}},
                }
            }
        },
        "domains": {
            "left": {
                "from_gmsh": {
                    "topology": "assembly",
                    "physical_group": "left",
                    "entity_kind": "volume",
                }
            },
            "right": {
                "from_gmsh": {
                    "topology": "assembly",
                    "physical_group": "right",
                    "entity_kind": "volume",
                }
            },
            "combined": {"union": ["left", "right"]},
            "exterior": {"boundary": "combined"},
            "left_exterior": {"exterior_cells": "left_component"},
            "left_interior": {"difference": ["left", "left_exterior"]},
        },
        "optical_components": {
            "left_component": {"domain": "left", "material": "yb_yag"},
            "right_component": {"domain": "right", "material": "yb_yag"},
        },
        "gain_media": {
            "amplifier": {"components": ["left_component", "right_component"]}
        },
        "simulation": {
            "optical_components": ["left_component", "right_component"],
            "gain_medium": "amplifier",
            "exterior_surface": "exterior",
            "initial_excitation": {"value": 0.2},
            "phi_ase": {"ase_steps": 1},
            "pumps": [
                {
                    "name": "seed",
                    "total_power": 1.0,
                    "ray_count": 32,
                    "pump_steps": 1,
                    "spectrum": {"monochromatic": 9.4e-7},
                    "injection": {"domain": "exterior"},
                }
            ],
            "time_integrator": {"method": "explicit_euler"},
            "time_step_size": 1.0e-6,
            "simulation_steps": 1,
        },
    }


def write(path, data=None):
    path.write_text(yaml.safe_dump(config() if data is None else data, sort_keys=False), encoding="utf-8")
    return path


def test_each_major_primitive_resolves_only_its_schema_v3_section(tmp_path):
    path = write(tmp_path / "simulation.yaml")
    resolved_material = material()

    domain = Domain.fromYaml(path, "combined")
    component = OpticalComponent.fromYaml(path, "left_component", materials={"yb_yag": resolved_material})
    medium = GainMedium.fromYaml(path, "amplifier", materials={"yb_yag": resolved_material})

    assert not domain.isEmpty
    assert component.material is resolved_material
    assert all(item.material is resolved_material for item in medium.components)


def test_simulation_from_yaml_reuses_injected_objects_and_owns_excitation(tmp_path):
    path = write(tmp_path / "simulation.yaml")
    resolved_material = material()

    simulation = Simulation.fromYaml(path, materials={"yb_yag": resolved_material})

    assert len(simulation.opticalComponents) == 2
    assert simulation.gainMedium.components == simulation.opticalComponents
    assert simulation.opticalComponents[0].material is resolved_material
    assert simulation.exteriorSurface.entityKind == "surface"
    assert not hasattr(simulation, "opticalDomain")
    assert simulation._backendGainMedium.get("betaVolume").value.tolist() == [0.2, 0.2]
    assert simulation.pump.sources[0].crossSections is simulation.crossSections
    assert not hasattr(simulation.pumps[0], "cross_sections")
    assert all(
        value.entityKind == "surface"
        for value in simulation._pumpRegistrations[0].injectionMethod.surface_domains
    )
    assert all(isinstance(value, int) for value in simulation.pump.sources[0].surfaceDomains)


def test_simulation_from_yaml_lowers_passive_component_outside_gain_medium(tmp_path):
    data = config()
    data["optical_components"]["right_component"]["material"] = "cladding"
    data["gain_media"]["amplifier"]["components"] = ["left_component"]
    path = write(tmp_path / "passive-component.yaml", data)
    gainMaterial = material()
    claddingMaterial = Material(
        materialName="passive cladding",
        temperature=293.15 * units.K,
        refractiveIndex=1.45,
        fluorescenceLifetime=None,
        crossSections=None,
        active=False,
        bulkAttenuation=5.5 / units.cm,
    )

    simulation = Simulation.fromYaml(
        path,
        materials={"yb_yag": gainMaterial, "cladding": claddingMaterial},
    )

    assert simulation.gainMedium.components == (simulation.opticalComponents[0],)
    assert simulation.opticalComponents[1].material is claddingMaterial
    assert simulation._backendGainMedium.get("claddingCellTypes").value.tolist() == [0, 1]
    assert simulation._backendGainMedium.get("claddingAbsorption").value == 5.5
    topology = simulation.opticalComponents[0].domain.topologies[0]
    assert not (
        simulation.exteriorSurface.maskFor(topology) & (topology.neighborCells >= 0)
    ).any()


def test_schema_v3_rejects_cycles_and_schema_v2(tmp_path):
    data = config()
    data["domains"]["left"] = {"union": ["right"]}
    data["domains"]["right"] = {"union": ["left"]}
    path = write(tmp_path / "cycle.yaml", data)
    try:
        Domain.fromYaml(path, "left")
    except ValueError as error:
        assert "cyclic YAML reference" in str(error)
    else:
        raise AssertionError("cyclic domain references were accepted")

    data = config()
    data["schema_version"] = 2
    path = write(tmp_path / "old.yaml", data)
    try:
        Simulation.fromYaml(path, materials={"yb_yag": material()})
    except ValueError as error:
        assert "schema_version: 3" in str(error)
    else:
        raise AssertionError("schema-v2 configuration was accepted")
