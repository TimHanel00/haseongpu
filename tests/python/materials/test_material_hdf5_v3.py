import numpy as np
import pytest
import yaml

from hase_units import units
from material_library import (
    Material,
    MaterialLibrary,
    TemperatureInterpolationWarning,
    loadBuiltinMaterials,
)

from test_physical_api import material


def test_builtin_library_resolves_mutable_material_with_recorded_units():
    resolved = loadBuiltinMaterials().resolve(
        "YbYAG",
        temperature=293.15 * units.K,
        activeIonDensity=2.776e20 / units.cm**3,
    )

    assert isinstance(resolved, Material)
    assert resolved.temperature.toValue(units.K) == pytest.approx(293.15)
    assert resolved.bulkAttenuation is None
    assert resolved.crossSections.wavelengths.unit.symbol == "nm"
    resolved.refractiveIndex = 1.9
    assert resolved.validate().refractiveIndex == pytest.approx(1.9)


def test_resolved_material_hdf5_roundtrip_preserves_units(tmp_path):
    source = material()
    path = tmp_path / "material.h5"
    source.toHdf5(path, key="Crystal")

    restored = Material.fromHdf5(
        path,
        key="Crystal",
        temperature=293.15 * units.K,
        activeIonDensity=source.activeIonDensity,
    )

    assert restored.materialName == source.materialName
    assert restored.bulkAttenuation is None
    assert restored.fluorescenceLifetime.toValue(units.ms) == pytest.approx(0.941)
    np.testing.assert_array_equal(
        restored.crossSections.emission.toValue(units.cm**2),
        source.crossSections.emission.toValue(units.cm**2),
    )


def test_material_library_rejects_duplicate_keys(tmp_path):
    library = MaterialLibrary()
    library.register("Crystal", material())
    with pytest.raises(KeyError, match="already registered"):
        library.register("Crystal", material())

    path = tmp_path / "library.h5"
    library.toHdf5(path)
    assert tuple(MaterialLibrary.fromHdf5(path)) == ("Crystal",)


def test_material_library_registers_and_roundtrips_temperature_states(tmp_path):
    roomTemperature = material()
    elevatedTemperature = material()
    elevatedTemperature.temperature = 303.15 * units.K
    elevatedTemperature.refractiveIndex = 1.85

    library = MaterialLibrary()
    library.register("Crystal", roomTemperature)
    assert library.registerState("Crystal", elevatedTemperature) is elevatedTemperature
    path = tmp_path / "temperature-states.h5"
    library.toHdf5(path)

    restored = MaterialLibrary.fromHdf5(path)
    with pytest.warns(TemperatureInterpolationWarning):
        interpolated = restored.resolve(
            "Crystal",
            temperature=298.15 * units.K,
            interpolation="linear",
        )
    assert interpolated.refractiveIndex == pytest.approx(1.84)


def test_passive_material_roundtrip_preserves_bulk_attenuation(tmp_path):
    source = Material(
        materialName="passive cladding",
        temperature=293.15 * units.K,
        refractiveIndex=1.45,
        fluorescenceLifetime=None,
        crossSections=None,
        bulkAttenuation=5.5 / units.cm,
    )
    path = tmp_path / "passive-material.h5"
    source.toHdf5(path, key="Cladding")

    restored = Material.fromHdf5(
        path,
        key="Cladding",
        temperature=293.15 * units.K,
    )

    assert restored.isPassive
    assert restored.bulkAttenuation.toValue(units.cm**-1) == pytest.approx(5.5)


def test_material_from_yaml_resolves_relative_hdf5_path(tmp_path):
    material().toHdf5(tmp_path / "materials.h5", key="YbYAG")
    config = {
        "schema_version": 3,
        "materials": {
            "yb_yag": {
                "from_hdf5": {"path": "materials.h5", "key": "YbYAG"},
                "temperature": 293.15,
                "active_ion_density": 2.776e20,
                "interpolation": "exact",
                "spectral_resolution": 1,
            }
        },
    }
    path = tmp_path / "simulation.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    resolved = Material.fromYaml(path, "yb_yag")

    assert resolved.name == "yb_yag"
    assert resolved.activeIonDensity.toValue(units.cm**-3) == pytest.approx(2.776e20)


def test_material_yaml_accepts_absorption_coefficient_alias(tmp_path):
    passive = Material(
        materialName="passive cladding",
        temperature=293.15 * units.K,
        refractiveIndex=1.45,
        fluorescenceLifetime=None,
        crossSections=None,
    )
    passive.toHdf5(tmp_path / "materials.h5", key="Cladding")
    config = {
        "schema_version": 3,
        "materials": {
            "cladding": {
                "from_hdf5": {"path": "materials.h5", "key": "Cladding"},
                "temperature": 293.15,
                "absorption_coefficient": 5.5,
            }
        },
    }
    path = tmp_path / "simulation.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    resolved = Material.fromYaml(path, "cladding")

    assert resolved.bulkAttenuation.toValue(units.cm**-1) == pytest.approx(5.5)

    config["materials"]["cladding"]["bulk_attenuation"] = 5.5
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="both bulk_attenuation and absorption_coefficient"):
        Material.fromYaml(path, "cladding")
