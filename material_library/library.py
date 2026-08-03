# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Material database container and persistence boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from .model import Material


class MaterialLibrary(Mapping):
    """Mapping of material keys to versioned physical material data.

    Lookup is deliberately mapping-only: use ``library["YbYAG"]``. Keys must
    be valid Python identifiers, but attribute access is not provided because
    misspellings should fail as explicit missing keys.
    """

    def __init__(self):
        self._materials: dict[str, Material] = {}

    def register(self, key: str, material: Material):
        """Register a material under a unique Python-identifier key.

        Parameters
        ----------
        key
            Stable database key such as ``"YbYAG"``.
        material
            :class:`Material` containing at least the states desired by the
            caller; empty materials are allowed during assembly.

        Returns
        -------
        Material
            The registered object, enabling inline construction.
        """
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError("material key must be a valid Python identifier")
        if not isinstance(material, Material):
            raise TypeError("material must be Material")
        if key in self._materials:
            raise KeyError(f"material key {key!r} is already registered")
        self._materials[key] = material
        return material

    def __getitem__(self, key):
        return self._materials[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._materials)

    def __len__(self):
        return len(self._materials)

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
            File produced by :meth:`toHdf5` or ``hase-material-convert``.
        """
        from .hdf5 import read_material_library

        library = cls()
        for key, material in read_material_library(path):
            library.register(key, material)
        return library
