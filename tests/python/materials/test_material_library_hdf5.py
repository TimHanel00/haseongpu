# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

import h5py
import numpy as np
import pytest
import warnings

from material_library import (
    CrossSectionTable,
    DATABASE_PATH,
    LegacyMaterialTextWarning,
    Material,
    MaterialLibrary,
    TemperatureInterpolationWarning,
    loadBuiltinMaterials,
)
from material_library.hdf5 import FORMAT_NAME, FORMAT_VERSION
from hase_units import units
from material_library.cli import main as convert_text_material


def spectra(scale=1.0):
    return CrossSectionTable(
        wavelengths=np.asarray([900.0, 1030.0]) * units.nm,
        absorption=np.asarray([1.1e-21, 1.2e-21]) * scale * units.cm**2,
        emission=np.asarray([2.0e-20, 2.48e-20]) * scale * units.cm**2,
        metadata={"measurement": "test spectrum"},
    )


def material_with_two_temperatures():
    return (
        Material("test crystal", metadata={"doi": "10.example/test"})
        .addState(
            temperature=280 * units.K,
            refractiveIndex=1.80,
            fluorescenceLifetime=1.0 * units.ms,
            crossSections=spectra(1.0),
            bulkAttenuation=0.1 / units.cm,
            metadata={"label": "cold"},
        )
        .addState(
            temperature=320 * units.K,
            refractiveIndex=1.84,
            fluorescenceLifetime=0.8 * units.ms,
            crossSections=spectra(2.0),
            bulkAttenuation=0.3 / units.cm,
            metadata={"label": "warm"},
        )
    )


def backend_linear_interpolation(values, wavelengths, sample_count):
    """Direct Python transcription of the former native interpolation loop."""

    values = [float(value) for value in values]
    wavelengths = [float(value) for value in wavelengths]
    if len(values) == 1:
        return np.full(sample_count, values[0], dtype=np.float64)
    target = [
        wavelengths[0]
        + (float(index) * (wavelengths[-1] - wavelengths[0])) / float(sample_count - 1)
        for index in range(sample_count)
    ]
    result = [0.0] * sample_count
    output_index = 0
    for index in range(len(wavelengths) - 1):
        slope = (values[index + 1] - values[index]) / (
            wavelengths[index + 1] - wavelengths[index]
        )
        intercept = values[index] - slope * wavelengths[index]
        while output_index < sample_count and target[output_index] < wavelengths[index + 1]:
            result[output_index] = intercept + slope * target[output_index]
            output_index += 1
    result[-1] = values[-1]
    return np.asarray(result)


def test_cross_sections_resample_with_native_linear_semantics():
    table = CrossSectionTable(
        wavelengths=np.asarray([900.0, 903.0, 910.0, 925.0]) * units.nm,
        absorption=np.asarray([0.01, 0.04, 0.015, 0.03]) * units.cm**2,
        emission=np.asarray([0.05, 0.02, 0.045, 0.01]) * units.cm**2,
    )

    result = table.resampleLinear(1000)

    expected_wavelengths = backend_linear_interpolation(
        table.wavelengths.magnitude,
        table.wavelengths.magnitude,
        1000,
    )
    expected_absorption = backend_linear_interpolation(
        table.absorption.magnitude,
        table.wavelengths.magnitude,
        1000,
    )
    expected_emission = backend_linear_interpolation(
        table.emission.magnitude,
        table.wavelengths.magnitude,
        1000,
    )
    np.testing.assert_array_equal(result.wavelengths.magnitude, expected_wavelengths)
    np.testing.assert_array_equal(result.absorption.magnitude, expected_absorption)
    np.testing.assert_array_equal(result.emission.magnitude, expected_emission)
    assert result.metadata["spectral_resampling"] == {
        "method": "linear",
        "source_sample_count": 4,
        "sample_count": 1000,
    }


def test_cross_sections_resampling_repeats_monochromatic_data():
    table = CrossSectionTable.monochromatic(
        wavelength=1030 * units.nm,
        absorption=1.0e-21 * units.cm**2,
        emission=2.0e-20 * units.cm**2,
    )

    result = table.resampleLinear(4)

    np.testing.assert_array_equal(result.wavelengths.magnitude, [1030.0] * 4)
    np.testing.assert_array_equal(result.absorption.magnitude, [1.0e-21] * 4)
    np.testing.assert_array_equal(result.emission.magnitude, [2.0e-20] * 4)


def test_cross_sections_resampling_rejects_downsampling():
    with pytest.raises(ValueError, match="greater than or equal"):
        spectra().resampleLinear(1)


def test_material_condition_applies_spectral_resolution_after_temperature_selection():
    condition = material_with_two_temperatures().at(
        temperature=280 * units.K,
        activeIonDensity=1.0e20 / units.cm**3,
        spectralResolution=16,
    )

    assert condition.crossSections.wavelengths.size == 16
    assert condition.crossSections.metadata["spectral_resampling"]["source_sample_count"] == 2


def test_material_library_hdf5_roundtrip_preserves_units_and_metadata(tmp_path):
    library = MaterialLibrary()
    library.register("TestCrystal", material_with_two_temperatures())
    filename = tmp_path / "materials.h5"

    library.toHdf5(filename)
    restored = MaterialLibrary.fromHdf5(filename)

    assert tuple(restored) == ("TestCrystal",)
    material = restored["TestCrystal"]
    assert material.name == "test crystal"
    assert material.metadata == {"doi": "10.example/test"}
    assert len(material.states) == 2
    cold = material.states[0]
    assert cold.temperature.unit.symbol == "K"
    assert cold.temperature.toValue(units.K) == pytest.approx(280.0)
    assert cold.fluorescenceLifetime.unit.symbol == "ms"
    assert cold.bulkAttenuation.unit.symbol == "cm^-1"
    assert cold.crossSections.wavelengths.unit.symbol == "nm"
    assert cold.crossSections.absorption.unit.symbol == "cm^2"
    np.testing.assert_array_equal(cold.crossSections.absorption.magnitude, [1.1e-21, 1.2e-21])
    assert cold.crossSections.metadata == {"measurement": "test spectrum"}

    with h5py.File(filename, "r") as handle:
        assert handle.attrs["format"] == FORMAT_NAME
        assert handle.attrs["format_version"] == FORMAT_VERSION
        wavelength = handle["materials/TestCrystal/states/0000/cross_sections/wavelength"]
        assert wavelength.attrs["unit"] == "nm"
        assert wavelength.attrs["unitSI"] == pytest.approx(1.0e-9)
        np.testing.assert_array_equal(
            wavelength.attrs["unitDimension"],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )


def test_material_can_be_written_and_read_without_a_library_wrapper(tmp_path):
    filename = tmp_path / "one-material.h5"
    material_with_two_temperatures().toHdf5(filename, key="Crystal")

    restored = Material.fromHdf5(filename)

    assert restored.name == "test crystal"
    assert len(restored.states) == 2


def test_multiple_materials_share_one_hdf5_database(tmp_path):
    library = MaterialLibrary()
    library.register("First", material_with_two_temperatures())
    library.register(
        "Second",
        Material("window").addState(
            temperature=300 * units.K,
            refractiveIndex=1.5,
            metadata={"source": "test"},
        ),
    )
    filename = tmp_path / "database.h5"

    library.toHdf5(filename)
    restored = MaterialLibrary.fromHdf5(filename)

    assert tuple(restored) == ("First", "Second")
    assert restored["Second"].at(temperature=300 * units.K).isTransparent


def test_material_library_requires_indexed_lookup():
    library = MaterialLibrary()
    library.register("TestCrystal", material_with_two_temperatures())

    assert library["TestCrystal"].name == "test crystal"
    with pytest.raises(AttributeError):
        _ = library.TestCrystal


def test_linear_temperature_interpolation_warns_with_bounding_temperatures():
    material = material_with_two_temperatures()

    with pytest.warns(
        TemperatureInterpolationWarning,
        match=r"no test crystal material state exists at 300 K; "
        r"interpolating between 280 K and 320 K",
    ):
        state = material.at(temperature=300 * units.K, interpolation="linear")

    assert state.temperature.toValue(units.K) == pytest.approx(300.0)
    assert state.refractiveIndex == pytest.approx(1.82)
    assert state.fluorescenceLifetime.toValue(units.ms) == pytest.approx(0.9)
    assert state.bulkAttenuation.toValue(units.cm**-1) == pytest.approx(0.2)
    np.testing.assert_allclose(
        state.crossSections.absorption.toValue(units.cm**2),
        [1.65e-21, 1.8e-21],
    )


def test_exact_temperature_does_not_warn():
    material = material_with_two_temperatures()

    with warnings.catch_warnings():
        warnings.simplefilter("error", TemperatureInterpolationWarning)
        state = material.at(temperature=280 * units.K, interpolation="linear")

    assert state.refractiveIndex == pytest.approx(1.80)


def test_temperature_extrapolation_is_rejected():
    with pytest.raises(ValueError, match="extrapolation"):
        material_with_two_temperatures().at(
            temperature=340 * units.K,
            interpolation="linear",
        )


def test_unknown_temperature_interpolation_mode_is_rejected_even_at_exact_state():
    with pytest.raises(ValueError, match="'exact' or 'linear'"):
        material_with_two_temperatures().at(
            temperature=280 * units.K,
            interpolation="cubic",
        )


def test_legacy_text_import_is_explicitly_discouraged(tmp_path):
    np.savetxt(tmp_path / "lambda_a.txt", [900.0, 1000.0])
    np.savetxt(tmp_path / "lambda_e.txt", [900.0, 1000.0])
    np.savetxt(tmp_path / "sigma_a.txt", [1.0, 2.0])
    np.savetxt(tmp_path / "sigma_e.txt", [3.0, 4.0])

    with pytest.warns(LegacyMaterialTextWarning, match="convert the material to HDF5"):
        table = CrossSectionTable.fromTextDirectory(tmp_path)

    assert table.wavelengths.unit.symbol == "nm"
    assert table.absorption.unit.symbol == "cm^2"
    assert table.metadata["source_format"] == "legacy-four-file-text"


def test_builtin_yb_yag_cross_sections_are_recorded_at_room_temperature():
    library = loadBuiltinMaterials()

    assert DATABASE_PATH.suffix == ".h5"
    state = library["YbYAG"].states[0]
    assert state.temperature.toValue(units.K) == pytest.approx(293.15)
    assert state.temperature.unit.symbol == "K"
    assert state.metadata["temperature_context"] == (
        "cross sections recorded at room temperature"
    )
    condition = library["YbYAG"].at(temperature=293.15 * units.K)
    assert condition.temperature.toValue(units.K) == pytest.approx(293.15)
    with h5py.File(DATABASE_PATH, "r") as handle:
        temperature = handle["materials/YbYAG/states/0000/temperature"]
        assert temperature[()] == pytest.approx(293.15)
        assert temperature.attrs["unit"] == "K"
        assert temperature.attrs["unitSI"] == pytest.approx(1.0)
        np.testing.assert_array_equal(
            temperature.attrs["unitDimension"],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )


def test_text_conversion_utility_creates_readable_hdf5(tmp_path):
    source = tmp_path / "text"
    source.mkdir()
    np.savetxt(source / "lambda_a.txt", [900.0, 1000.0])
    np.savetxt(source / "lambda_e.txt", [900.0, 1000.0])
    np.savetxt(source / "sigma_a.txt", [1.0e-21, 2.0e-21])
    np.savetxt(source / "sigma_e.txt", [3.0e-21, 4.0e-21])
    target = tmp_path / "converted.h5"

    with pytest.warns(LegacyMaterialTextWarning):
        convert_text_material(
            [
                str(source),
                str(target),
                "--key",
                "TestMaterial",
                "--name",
                "test material",
                "--refractive-index",
                "1.7",
                "--fluorescence-lifetime-seconds",
                "0.002",
                "--temperature-kelvin",
                "295",
                "--source",
                "test fixture",
            ]
        )

    restored = MaterialLibrary.fromHdf5(target)["TestMaterial"]
    state = restored.at(temperature=295 * units.K)
    assert state.refractiveIndex == pytest.approx(1.7)
    assert state.fluorescenceLifetime.toValue(units.ms) == pytest.approx(2.0)
    assert state.metadata["source"] == "test fixture"
