# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Versioned HDF5 codec for the standalone material library."""

from __future__ import annotations

import json

import h5py
import numpy as np

from hase_units import Quantity, Unit, units
from .model import CrossSectionTable, _MaterialRecord, _MaterialState


FORMAT_NAME = "HASEonGPU-material-library"
FORMAT_VERSION = "1.0"


def _write_metadata(owner, metadata):
    owner.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)


def _read_metadata(owner):
    value = owner.attrs.get("metadata_json", "{}")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = json.loads(str(value))
    if not isinstance(result, dict):
        raise ValueError("material metadata_json must decode to an object")
    return result


def _write_quantity(group, name, quantity):
    dataset = group.create_dataset(name, data=np.asarray(quantity.magnitude))
    dataset.attrs["unit"] = quantity.unit.symbol
    dataset.attrs["unitSI"] = quantity.unit.unitSI
    dataset.attrs["unitDimension"] = np.asarray(quantity.unit.unitDimension, dtype=np.float64)
    return dataset


def _read_quantity(group, name):
    dataset = group[name]
    symbol = dataset.attrs["unit"]
    if isinstance(symbol, bytes):
        symbol = symbol.decode("utf-8")
    unit = Unit.fromOpenPmd(
        symbol,
        float(dataset.attrs["unitSI"]),
        tuple(np.asarray(dataset.attrs["unitDimension"], dtype=np.float64)),
    )
    values = dataset[()]
    return Quantity(values.item() if np.ndim(values) == 0 else values, unit)


def write_material_library(path, entries, *, overwrite=False):
    mode = "w" if overwrite else "x"
    with h5py.File(path, mode) as handle:
        handle.attrs["format"] = FORMAT_NAME
        handle.attrs["format_version"] = FORMAT_VERSION
        materials_group = handle.create_group("materials", track_order=True)
        for key, material in entries:
            material_group = materials_group.create_group(key, track_order=True)
            material_group.attrs["name"] = material.name
            _write_metadata(material_group, material.metadata)
            states_group = material_group.create_group("states", track_order=True)
            for index, state in enumerate(material.states):
                state_group = states_group.create_group(f"{index:04d}")
                state_group.attrs["temperature_known"] = state.temperature is not None
                if state.temperature is not None:
                    _write_quantity(state_group, "temperature", state.temperature)
                _write_quantity(
                    state_group,
                    "refractive_index",
                    Quantity(state.refractiveIndex, units.one),
                )
                if state.fluorescenceLifetime is not None:
                    _write_quantity(
                        state_group, "fluorescence_lifetime", state.fluorescenceLifetime
                    )
                if state.bulkAttenuation is not None:
                    _write_quantity(state_group, "bulk_attenuation", state.bulkAttenuation)
                _write_metadata(state_group, state.metadata)
                if state.crossSections is not None:
                    spectra = state_group.create_group("cross_sections")
                    _write_quantity(spectra, "wavelength", state.crossSections.wavelengths)
                    _write_quantity(spectra, "absorption", state.crossSections.absorption)
                    _write_quantity(spectra, "emission", state.crossSections.emission)
                    _write_metadata(spectra, state.crossSections.metadata)


def read_material_library(path):
    with h5py.File(path, "r") as handle:
        format_name = handle.attrs.get("format")
        format_version = handle.attrs.get("format_version")
        if isinstance(format_name, bytes):
            format_name = format_name.decode("utf-8")
        if isinstance(format_version, bytes):
            format_version = format_version.decode("utf-8")
        if format_name != FORMAT_NAME:
            raise ValueError(f"not a {FORMAT_NAME} file")
        if format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported material-library format version {format_version!r}; "
                f"expected {FORMAT_VERSION!r}"
            )
        result = []
        for key, material_group in handle["materials"].items():
            name = material_group.attrs["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            material = _MaterialRecord(str(name), metadata=_read_metadata(material_group))
            for state_group in material_group["states"].values():
                temperature = (
                    _read_quantity(state_group, "temperature")
                    if bool(state_group.attrs["temperature_known"])
                    else None
                )
                spectra = None
                if "cross_sections" in state_group:
                    spectra_group = state_group["cross_sections"]
                    spectra = CrossSectionTable(
                        wavelengths=_read_quantity(spectra_group, "wavelength"),
                        absorption=_read_quantity(spectra_group, "absorption"),
                        emission=_read_quantity(spectra_group, "emission"),
                        metadata=_read_metadata(spectra_group),
                    )
                state = _MaterialState(
                    temperature=temperature,
                    refractiveIndex=float(
                        _read_quantity(state_group, "refractive_index").toValue(units.one)
                    ),
                    fluorescenceLifetime=(
                        _read_quantity(state_group, "fluorescence_lifetime")
                        if "fluorescence_lifetime" in state_group
                        else None
                    ),
                    crossSections=spectra,
                    bulkAttenuation=(
                        _read_quantity(state_group, "bulk_attenuation")
                        if "bulk_attenuation" in state_group
                        else None
                    ),
                    metadata=_read_metadata(state_group),
                )
                material._append_state(state)
            result.append((str(key), material))
        return result
