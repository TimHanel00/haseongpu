# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Topology-independent optical material definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossSectionTable:
    """Isotropic absorption and emission cross sections on one wavelength grid.

    All public values use SI units: wavelengths are metres and cross sections
    are square metres.
    """

    wavelengths: object
    absorption: object
    emission: object

    def __post_init__(self):
        wavelengths = np.asarray(self.wavelengths, dtype=np.float64).reshape(-1).copy()
        absorption = np.asarray(self.absorption, dtype=np.float64).reshape(-1).copy()
        emission = np.asarray(self.emission, dtype=np.float64).reshape(-1).copy()
        if wavelengths.size == 0:
            raise ValueError("cross-section wavelengths must not be empty")
        if wavelengths.size != absorption.size or wavelengths.size != emission.size:
            raise ValueError("wavelength, absorption, and emission arrays must have the same length")
        if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
            raise ValueError("cross-section wavelengths must be finite and positive")
        if np.any(np.diff(wavelengths) <= 0.0):
            raise ValueError("cross-section wavelengths must be strictly increasing")
        if np.any(~np.isfinite(absorption)) or np.any(absorption < 0.0):
            raise ValueError("absorption cross sections must be finite and non-negative")
        if np.any(~np.isfinite(emission)) or np.any(emission < 0.0):
            raise ValueError("emission cross sections must be finite and non-negative")
        wavelengths.flags.writeable = False
        absorption.flags.writeable = False
        emission.flags.writeable = False
        object.__setattr__(self, "wavelengths", wavelengths)
        object.__setattr__(self, "absorption", absorption)
        object.__setattr__(self, "emission", emission)

    @classmethod
    def monochromatic(cls, *, wavelength, absorption, emission):
        return cls([wavelength], [absorption], [emission])

    @classmethod
    def from_directory(cls, path):
        """Load the four historical text columns and convert nm/cm² to SI."""
        from pathlib import Path

        root = Path(path)
        wavelengths_absorption = np.loadtxt(root / "lambda_a.txt", dtype=np.float64).reshape(-1) * 1.0e-9
        wavelengths_emission = np.loadtxt(root / "lambda_e.txt", dtype=np.float64).reshape(-1) * 1.0e-9
        absorption = np.loadtxt(root / "sigma_a.txt", dtype=np.float64).reshape(-1) * 1.0e-4
        emission = np.loadtxt(root / "sigma_e.txt", dtype=np.float64).reshape(-1) * 1.0e-4
        wavelengths = np.unique(np.concatenate((wavelengths_absorption, wavelengths_emission)))
        return cls(
            wavelengths,
            np.interp(wavelengths, wavelengths_absorption, absorption, left=0.0, right=0.0),
            np.interp(wavelengths, wavelengths_emission, emission, left=0.0, right=0.0),
        )


@dataclass(frozen=True)
class MaterialDefinition:
    """Reusable concentration-independent material physics in SI units."""

    name: str
    refractive_index: float
    fluorescence_lifetime: float | None = None
    cross_sections: CrossSectionTable | None = None
    bulk_attenuation: float = 0.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("material definition name must be a non-empty string")
        if not np.isfinite(self.refractive_index) or self.refractive_index <= 0.0:
            raise ValueError("material refractive_index must be finite and positive")
        if self.fluorescence_lifetime is not None and (
            not np.isfinite(self.fluorescence_lifetime) or self.fluorescence_lifetime <= 0.0
        ):
            raise ValueError("material fluorescence_lifetime must be finite and positive")
        if self.cross_sections is not None and not isinstance(self.cross_sections, CrossSectionTable):
            raise TypeError("material cross_sections must be CrossSectionTable or None")
        if not np.isfinite(self.bulk_attenuation) or self.bulk_attenuation < 0.0:
            raise ValueError("material bulk_attenuation must be finite and non-negative")


@dataclass(frozen=True, eq=False)
class MaterialInstance:
    """Run-specific instance of a reusable material definition."""

    definition: MaterialDefinition
    active_ion_density: float = 0.0
    name: str | None = None
    optical_axis: object | None = None

    def __post_init__(self):
        if not isinstance(self.definition, MaterialDefinition):
            raise TypeError("material instance definition must be MaterialDefinition")
        if not np.isfinite(self.active_ion_density) or self.active_ion_density < 0.0:
            raise ValueError("active_ion_density must be finite and non-negative")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("material instance name must be a non-empty string or None")
        if self.optical_axis is not None:
            axis = np.asarray(self.optical_axis, dtype=np.float64).reshape(-1)
            if axis.shape != (3,) or np.any(~np.isfinite(axis)) or np.linalg.norm(axis) == 0.0:
                raise ValueError("optical_axis must be a finite non-zero three-vector")
            object.__setattr__(self, "optical_axis", tuple(axis / np.linalg.norm(axis)))
        if self.active_ion_density > 0.0:
            if self.definition.fluorescence_lifetime is None:
                raise ValueError("active materials require fluorescence_lifetime")
            if self.definition.cross_sections is None:
                raise ValueError("active materials require cross_sections")
            if not np.any(self.definition.cross_sections.emission > 0.0):
                raise ValueError("active materials require a non-zero emission cross section")

    @property
    def display_name(self):
        return self.name or self.definition.name

    @property
    def is_active(self):
        return self.active_ion_density > 0.0

    @property
    def is_passive(self):
        return not self.is_active and self.definition.bulk_attenuation > 0.0

    @property
    def is_transparent(self):
        return not self.is_active and self.definition.bulk_attenuation == 0.0
