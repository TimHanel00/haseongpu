# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Material database container and persistence boundary."""

from __future__ import annotations

from collections.abc import Iterator

from .model import Material, _MaterialRecord, _MaterialState


class MaterialLibrary:
    """Versioned material records resolved into mutable :class:`Material` values."""

    def __init__(self):
        self._materials: dict[str, _MaterialRecord] = {}

    def register(self, key: str, material):
        """Register a material under a unique Python-identifier key.

        Parameters
        ----------
        key
            Stable database key such as ``"YbYAG"``.
        material
            Resolved :class:`Material` providing the first temperature state.

        Returns
        -------
        Material
            The registered object, enabling inline construction.
        """
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError("material key must be a valid Python identifier")
        if not isinstance(material, Material):
            raise TypeError("material must be a resolved Material")
        if key in self._materials:
            raise KeyError(f"material key {key!r} is already registered")
        material.validate()
        record = _MaterialRecord(material.materialName, metadata=material.metadata)
        record._append_state(self._stateFromMaterial(material))
        self._materials[key] = record
        return material

    def registerState(self, key: str, material):
        """Add another resolved temperature state to an existing material key."""
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError("material key must be a valid Python identifier")
        if not isinstance(material, Material):
            raise TypeError("material must be a resolved Material")
        try:
            record = self._materials[key]
        except KeyError as exc:
            raise KeyError(f"unknown material key {key!r}") from exc
        material.validate()
        if material.materialName != record.name:
            raise ValueError("temperature states under one key must have the same materialName")
        record._append_state(self._stateFromMaterial(material))
        return material

    @staticmethod
    def _stateFromMaterial(material):
        return _MaterialState(
            temperature=material.temperature,
            refractiveIndex=material.refractiveIndex,
            fluorescenceLifetime=material.fluorescenceLifetime,
            crossSections=material.crossSections,
            bulkAttenuation=material.bulkAttenuation,
            metadata=material.metadata,
        )

    def _registerRecord(self, key, record):
        if key in self._materials:
            raise KeyError(f"material key {key!r} is already registered")
        self._materials[key] = record

    def __iter__(self) -> Iterator[str]:
        return iter(self._materials)

    def __len__(self):
        return len(self._materials)

    def resolve(self, key, **condition):
        """Resolve ``key`` into a new mutable :class:`Material`."""
        try:
            record = self._materials[str(key)]
        except KeyError as exc:
            raise KeyError(f"unknown material key {key!r}") from exc
        return record.at(**condition)

    def toHdf5(self, path, *, overwrite=False):
        """Write every registered material to a versioned HDF5 file.

        Parameters
        ----------
        path
            Destination file path.
        overwrite
            Replace an existing file only when explicitly true.
        """
        from .hdf5 import write_material_library

        write_material_library(path, self._materials.items(), overwrite=overwrite)

    @classmethod
    def fromHdf5(cls, path):
        """Load and validate a complete versioned HDF5 material library.

        Parameters
        ----------
        path
            File produced by :meth:`toHdf5`.
        """
        from .hdf5 import read_material_library

        library = cls()
        for key, material in read_material_library(path):
            library._registerRecord(key, material)
        return library
