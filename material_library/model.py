# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Temperature-resolved optical material data independent of simulation code."""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

from hase_units import LENGTH, TEMPERATURE, TIME, Quantity, requireQuantity, units


AREA = tuple(2.0 * exponent for exponent in LENGTH)
INVERSE_LENGTH = tuple(-exponent for exponent in LENGTH)
NUMBER_DENSITY = tuple(-3.0 * exponent for exponent in LENGTH)


class LegacyMaterialTextWarning(UserWarning):
    """Warn that a legacy four-file spectrum lacks structured metadata."""


class TemperatureInterpolationWarning(UserWarning):
    """Warn that no measured state exists at the requested temperature."""


class LegacyMaterialActivityWarning(UserWarning):
    """Warn that an HDF5 v1.0 material lacks an explicit activity flag."""


def _metadata_copy(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    result = dict(metadata)
    if not all(isinstance(key, str) and key for key in result):
        raise ValueError("metadata keys must be non-empty strings")
    return result


@dataclass(frozen=True)
class CrossSectionTable:
    """Absorption and emission cross sections on one wavelength grid.

    Parameters
    ----------
    wavelengths
        Non-empty one-dimensional positive length quantity. Values must be
        strictly increasing, except a repeated constant is accepted for
        explicitly monochromatic resampling.
    absorption, emission
        Finite non-negative area quantities with the same shape as
        ``wavelengths``.
    metadata
        Optional string-keyed provenance and measurement metadata.
    """

    wavelengths: Quantity
    """Strictly increasing positive wavelength grid shared by both curves."""
    absorption: Quantity
    """Non-negative absorption area per active ion at each wavelength."""
    emission: Quantity
    """Non-negative stimulated-emission area per active ion at each wavelength."""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """String-keyed measurement provenance and processing history."""

    def __post_init__(self):
        wavelengths = requireQuantity(self.wavelengths, LENGTH, "cross-section wavelengths")
        absorption = requireQuantity(self.absorption, AREA, "absorption cross sections")
        emission = requireQuantity(self.emission, AREA, "emission cross sections")
        wavelengths_si = np.asarray(wavelengths.toValue(units.m), dtype=np.float64).reshape(-1)
        absorption_si = np.asarray(absorption.toValue(units.m**2), dtype=np.float64).reshape(-1)
        emission_si = np.asarray(emission.toValue(units.m**2), dtype=np.float64).reshape(-1)
        if wavelengths_si.size == 0:
            raise ValueError("cross-section wavelengths must not be empty")
        if wavelengths_si.size != absorption_si.size or wavelengths_si.size != emission_si.size:
            raise ValueError("wavelength, absorption, and emission arrays must have the same length")
        if np.any(~np.isfinite(wavelengths_si)) or np.any(wavelengths_si <= 0.0):
            raise ValueError("cross-section wavelengths must be finite and positive")
        wavelength_steps = np.diff(wavelengths_si)
        if np.any(wavelength_steps <= 0.0) and not np.all(wavelength_steps == 0.0):
            raise ValueError(
                "cross-section wavelengths must be strictly increasing or constant "
                "for monochromatic data"
            )
        if np.any(~np.isfinite(absorption_si)) or np.any(absorption_si < 0.0):
            raise ValueError("absorption cross sections must be finite and non-negative")
        if np.any(~np.isfinite(emission_si)) or np.any(emission_si < 0.0):
            raise ValueError("emission cross sections must be finite and non-negative")
        object.__setattr__(self, "wavelengths", Quantity(wavelengths.magnitude, wavelengths.unit))
        object.__setattr__(self, "absorption", Quantity(absorption.magnitude, absorption.unit))
        object.__setattr__(self, "emission", Quantity(emission.magnitude, emission.unit))
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    @classmethod
    def monochromatic(cls, *, wavelength, absorption, emission, metadata=None):
        """Construct a one-sample material spectrum.

        Parameters
        ----------
        wavelength
            Positive scalar length quantity.
        absorption, emission
            Non-negative scalar area quantities.
        metadata
            Optional spectrum provenance.
        """
        wavelength = requireQuantity(wavelength, LENGTH, "wavelength")
        absorption = requireQuantity(absorption, AREA, "absorption")
        emission = requireQuantity(emission, AREA, "emission")
        return cls(
            Quantity([wavelength.magnitude], wavelength.unit),
            Quantity([absorption.magnitude], absorption.unit),
            Quantity([emission.magnitude], emission.unit),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def fromTextDirectory(cls, path, *, metadata=None):
        """Import the historical nm/cm² four-file representation.

        Parameters
        ----------
        path
            Directory containing ``lambda_a.txt``, ``lambda_e.txt``,
            ``sigma_a.txt``, and ``sigma_e.txt``. Historical units are assumed
            to be nanometres and square centimetres.
        metadata
            Optional provenance merged with ``source_format``.

        Returns
        -------
        CrossSectionTable
            Union wavelength grid with each curve linearly interpolated and
            zero outside its original support.

        Warns
        -----
        LegacyMaterialTextWarning
            Always, because text files do not encode units or temperature.
        """

        warnings.warn(
            "four-file material spectra do not carry units, temperature, or provenance; "
            "convert the material to HDF5",
            LegacyMaterialTextWarning,
            stacklevel=2,
        )
        root = Path(path)
        wavelengths_absorption = np.loadtxt(root / "lambda_a.txt", dtype=np.float64).reshape(-1)
        wavelengths_emission = np.loadtxt(root / "lambda_e.txt", dtype=np.float64).reshape(-1)
        absorption = np.loadtxt(root / "sigma_a.txt", dtype=np.float64).reshape(-1)
        emission = np.loadtxt(root / "sigma_e.txt", dtype=np.float64).reshape(-1)
        wavelengths = np.unique(np.concatenate((wavelengths_absorption, wavelengths_emission)))
        imported_metadata = _metadata_copy(metadata)
        imported_metadata.setdefault("source_format", "legacy-four-file-text")
        return cls(
            Quantity(wavelengths, units.nm),
            Quantity(
                np.interp(wavelengths, wavelengths_absorption, absorption, left=0.0, right=0.0),
                units.cm**2,
            ),
            Quantity(
                np.interp(wavelengths, wavelengths_emission, emission, left=0.0, right=0.0),
                units.cm**2,
            ),
            metadata=imported_metadata,
        )

    def absorptionAt(self, wavelength):
        """Linearly interpolate absorption at a scalar wavelength quantity.

        Parameters
        ----------
        wavelength
            Scalar length quantity in any compatible unit.

        Returns
        -------
        Quantity
            Scalar in the table's absorption unit.
        """
        wavelength = requireQuantity(wavelength, LENGTH, "wavelength")
        value = np.interp(
            float(wavelength.toValue(self.wavelengths.unit)),
            np.asarray(self.wavelengths.magnitude),
            np.asarray(self.absorption.magnitude),
        )
        return Quantity(value, self.absorption.unit)

    def emissionAt(self, wavelength):
        """Linearly interpolate stimulated emission at a scalar wavelength.

        Parameters
        ----------
        wavelength
            Scalar length quantity in any compatible unit.

        Returns
        -------
        Quantity
            Scalar in the table's emission unit.
        """
        wavelength = requireQuantity(wavelength, LENGTH, "wavelength")
        value = np.interp(
            float(wavelength.toValue(self.wavelengths.unit)),
            np.asarray(self.wavelengths.magnitude),
            np.asarray(self.emission.magnitude),
        )
        return Quantity(value, self.emission.unit)

    def resampleLinear(self, spectralResolution):
        """Resample both curves onto an endpoint-inclusive uniform grid.

        Parameters
        ----------
        spectralResolution
            Integer target sample count. It must be at least the source count;
            downsampling is rejected because it would discard material data.

        Returns
        -------
        CrossSectionTable
            ``self`` when the count is unchanged, otherwise a new table on a
            uniform endpoint-inclusive grid. A one-wavelength source is
            repeated without inventing a wavelength interval.
        """

        if isinstance(spectralResolution, (bool, np.bool_)):
            raise TypeError("spectralResolution must be an integer")
        try:
            sample_count = operator.index(spectralResolution)
        except TypeError as exc:
            raise TypeError("spectralResolution must be an integer") from exc
        source_count = self.wavelengths.size
        if sample_count < source_count:
            raise ValueError(
                "spectralResolution must be greater than or equal to "
                f"the cross-section sample count ({source_count})"
            )
        if sample_count == source_count:
            return self

        wavelengths = np.asarray(self.wavelengths.magnitude, dtype=np.float64)
        absorption = np.asarray(self.absorption.magnitude, dtype=np.float64)
        emission = np.asarray(self.emission.magnitude, dtype=np.float64)
        if source_count == 1:
            target_wavelengths = np.full(sample_count, wavelengths[0], dtype=np.float64)
            target_absorption = np.full(sample_count, absorption[0], dtype=np.float64)
            target_emission = np.full(sample_count, emission[0], dtype=np.float64)
        else:
            target_wavelengths = wavelengths[0] + (
                np.arange(sample_count, dtype=np.float64) * (wavelengths[-1] - wavelengths[0])
            ) / float(sample_count - 1)
            segment = np.searchsorted(wavelengths[1:], target_wavelengths[:-1], side="right")

            def interpolate(values):
                slope = (values[segment + 1] - values[segment]) / (
                    wavelengths[segment + 1] - wavelengths[segment]
                )
                intercept = values[segment] - slope * wavelengths[segment]
                result = np.empty(sample_count, dtype=np.float64)
                result[:-1] = intercept + slope * target_wavelengths[:-1]
                result[-1] = values[-1]
                return result

            target_absorption = interpolate(absorption)
            target_emission = interpolate(emission)

        metadata = dict(self.metadata)
        metadata["spectral_resampling"] = {
            "method": "linear",
            "source_sample_count": source_count,
            "sample_count": sample_count,
        }
        return CrossSectionTable(
            Quantity(target_wavelengths, self.wavelengths.unit),
            Quantity(target_absorption, self.absorption.unit),
            Quantity(target_emission, self.emission.unit),
            metadata=metadata,
        )


@dataclass(frozen=True)
class _MaterialState:
    """Measured or modelled material properties at one temperature.

    Parameters
    ----------
    temperature
        Positive absolute-temperature quantity. Use ``None`` only when the
        source temperature is genuinely unknown and explain it with a
        non-empty ``metadata["temperature_status"]``.
    refractiveIndex
        Positive dimensionless interior optical index.
    fluorescenceLifetime
        Optional positive upper-state lifetime. Active material conditions
        require it.
    crossSections
        Optional wavelength-dependent absorption/emission table. Active
        material conditions require non-zero emission data.
    bulkAttenuation
        Optional non-negative passive attenuation coefficient with
        inverse-length unit. ``None`` selects no bulk-loss contribution.
    metadata
        String-keyed state provenance.
    """

    temperature: Quantity | None
    """Absolute temperature at which this physical state applies."""
    refractiveIndex: float
    """Positive dimensionless phase refractive index inside the material."""
    fluorescenceLifetime: Quantity | None = None
    """Upper-state spontaneous-decay lifetime required for active media."""
    crossSections: CrossSectionTable | None = None
    """Wavelength-dependent per-ion absorption and emission cross sections."""
    bulkAttenuation: Quantity | None = None
    """Optional passive intensity attenuation coefficient in inverse length."""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """String-keyed provenance specific to this temperature state."""

    def __post_init__(self):
        temperature = requireQuantity(
            self.temperature, TEMPERATURE, "material temperature", allowNone=True
        )
        if temperature is not None:
            value = float(temperature.toValue(units.K))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("material temperature must be finite and above 0 K")
        if not np.isfinite(self.refractiveIndex) or self.refractiveIndex <= 0.0:
            raise ValueError("material refractiveIndex must be finite and positive")
        lifetime = requireQuantity(
            self.fluorescenceLifetime,
            TIME,
            "material fluorescenceLifetime",
            allowNone=True,
        )
        if lifetime is not None:
            lifetime_si = float(lifetime.toValue(units.s))
            if not np.isfinite(lifetime_si) or lifetime_si <= 0.0:
                raise ValueError("material fluorescenceLifetime must be finite and positive")
        if self.crossSections is not None and not isinstance(self.crossSections, CrossSectionTable):
            raise TypeError("material crossSections must be CrossSectionTable or None")
        if self.bulkAttenuation is not None:
            attenuation = requireQuantity(
                self.bulkAttenuation, INVERSE_LENGTH, "material bulkAttenuation"
            )
            attenuation_si = float(attenuation.toValue(units.m**-1))
            if not np.isfinite(attenuation_si) or attenuation_si < 0.0:
                raise ValueError("material bulkAttenuation must be finite and non-negative")
        state_metadata = _metadata_copy(self.metadata)
        if temperature is None and not state_metadata.get("temperature_status"):
            raise ValueError(
                "a state without a numeric temperature must describe temperature_status in metadata"
            )
        object.__setattr__(self, "metadata", state_metadata)


@dataclass(eq=False, init=False)
class Material:
    """Mutable material properties resolved for one optical component.

    Parameters
    ----------
    materialName
        Name of the originating material-library record.
    temperature
        Selected absolute temperature, or ``None`` for an explicitly
        unknown-temperature reference.
    refractiveIndex, fluorescenceLifetime, crossSections
        Physical state selected from the material database.
    active
        Whether the material participates in active-ion population dynamics.
        This classification is independent of the selected ion density.
    bulkAttenuation
        Optional passive intensity attenuation coefficient. The name
        ``absorptionCoefficient`` aliases this property.
    activeIonDensity
        Non-negative total active-ion number density for the selected run.
    name
        Optional run-specific label overriding ``materialName`` for display.
    opticalAxis
        Optional finite non-zero three-vector; normalized on construction.
    metadata
        Merged material/state provenance and selection metadata.
    """

    materialName: str
    """Name of the physical material from which this condition was resolved."""
    temperature: Quantity | None
    """Selected absolute material temperature."""
    refractiveIndex: float
    """Dimensionless refractive index used by optical transport."""
    fluorescenceLifetime: Quantity | None
    """Upper-state spontaneous-decay lifetime used by the rate equations."""
    crossSections: CrossSectionTable | None
    """Absorption and stimulated-emission cross sections on the run grid."""
    active: bool
    """Whether population excitation is evaluated for this material."""
    bulkAttenuation: Quantity | None = None
    """Optional passive intensity attenuation coefficient in inverse length."""
    activeIonDensity: Quantity = Quantity(0.0, units.m**-3)
    """Total active-ion number density selected for the current run."""
    name: str | None = None
    """Optional run-specific label with no effect on physical coefficients."""
    opticalAxis: object | None = None
    """Optional normalized material orientation vector for anisotropic models."""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Merged material, state, interpolation, and resampling provenance."""

    def __init__(
        self,
        materialName,
        temperature,
        refractiveIndex,
        fluorescenceLifetime,
        crossSections,
        bulkAttenuation=None,
        activeIonDensity=Quantity(0.0, units.m**-3),
        name=None,
        opticalAxis=None,
        metadata=None,
        *,
        active,
        absorptionCoefficient=None,
    ):
        if bulkAttenuation is not None and absorptionCoefficient is not None:
            raise TypeError("provide either bulkAttenuation or absorptionCoefficient, not both")
        self.materialName = materialName
        self.temperature = temperature
        self.refractiveIndex = refractiveIndex
        self.fluorescenceLifetime = fluorescenceLifetime
        self.crossSections = crossSections
        self.active = active
        self.activeIonDensity = activeIonDensity
        self.bulkAttenuation = (
            absorptionCoefficient if bulkAttenuation is None else bulkAttenuation
        )
        self.name = name
        self.opticalAxis = opticalAxis
        self.metadata = {} if metadata is None else metadata
        self.__post_init__()

    def __post_init__(self):
        if not isinstance(self.active, (bool, np.bool_)):
            raise TypeError("material active must be a boolean")
        self.active = bool(self.active)
        density = requireQuantity(self.activeIonDensity, NUMBER_DENSITY, "activeIonDensity")
        density_si = float(density.toValue(units.m**-3))
        if not np.isfinite(density_si) or density_si < 0.0:
            raise ValueError("activeIonDensity must be finite and non-negative")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("material condition name must be a non-empty string or None")
        if self.opticalAxis is not None:
            axis = np.asarray(self.opticalAxis, dtype=np.float64).reshape(-1)
            if axis.shape != (3,) or np.any(~np.isfinite(axis)) or np.linalg.norm(axis) == 0.0:
                raise ValueError("opticalAxis must be a finite non-zero three-vector")
            self.opticalAxis = tuple(axis / np.linalg.norm(axis))
        if self.isPassive and density_si > 0.0:
            raise ValueError("passive materials require zero activeIonDensity")
        if self.isActive:
            if self.fluorescenceLifetime is None:
                raise ValueError("active materials require fluorescenceLifetime")
            if self.crossSections is None:
                raise ValueError("active materials require crossSections")
            if not np.any(np.asarray(self.crossSections.emission.magnitude) > 0.0):
                raise ValueError("active materials require a non-zero emission cross section")
        if self.bulkAttenuation is not None:
            attenuation = requireQuantity(
                self.bulkAttenuation, INVERSE_LENGTH, "material bulkAttenuation"
            )
            attenuation_si = float(attenuation.toValue(units.m**-1))
            if not np.isfinite(attenuation_si) or attenuation_si < 0.0:
                raise ValueError("material bulkAttenuation must be finite and non-negative")
        self.metadata = _metadata_copy(self.metadata)

    @property
    def displayName(self):
        """Run-specific ``name`` when present, otherwise ``materialName``."""
        return self.name or self.materialName

    @property
    def isActive(self):
        """Whether this material participates in population dynamics."""
        return self.active

    @property
    def isPassive(self):
        """Whether this material is excluded from population dynamics."""
        return not self.isActive

    @property
    def isTransparent(self):
        """Whether a passive condition contributes no volumetric attenuation."""
        return (
            self.isPassive
            and (
                self.bulkAttenuation is None
                or float(self.bulkAttenuation.toValue(units.m**-1)) == 0.0
            )
        )

    @property
    def absorptionCoefficient(self):
        """Read or write :attr:`bulkAttenuation` through its physical alias."""
        return self.bulkAttenuation

    @absorptionCoefficient.setter
    def absorptionCoefficient(self, value):
        self.bulkAttenuation = value

    def validate(self):
        """Revalidate this mutable material after direct API-side edits."""
        self.__post_init__()
        return self

    def toHdf5(self, path, *, key=None, overwrite=False):
        """Write this resolved material as a one-state HDF5 library."""
        from .library import MaterialLibrary

        library = MaterialLibrary()
        library.register(key or _default_key(self.displayName), self)
        library.toHdf5(path, overwrite=overwrite)

    @classmethod
    def fromHdf5(
        cls,
        path,
        *,
        key=None,
        temperature=None,
        activeIonDensity=Quantity(0.0, units.m**-3),
        spectralResolution=None,
        interpolation="exact",
        name=None,
        opticalAxis=None,
    ):
        """Resolve one mutable material from a versioned HDF5 library."""
        from .library import MaterialLibrary

        library = MaterialLibrary.fromHdf5(path)
        if key is None:
            keys = tuple(library)
            if len(keys) != 1:
                raise ValueError("material HDF5 contains multiple entries; specify key")
            key = keys[0]
        return library.resolve(
            key,
            temperature=temperature,
            activeIonDensity=activeIonDensity,
            spectralResolution=spectralResolution,
            interpolation=interpolation,
            name=name,
            opticalAxis=opticalAxis,
        )

    @classmethod
    def fromYaml(cls, filename, name, **objects):
        """Resolve one named schema-v3 material without constructing a simulation."""
        from pyInclude.configuration import objectFromYaml

        return objectFromYaml(cls, filename, name, **objects)


class _MaterialRecord:
    """Named material with one or more temperature-resolved states.

    Parameters
    ----------
    name
        Non-empty physical material name.
    active
        Material-level active/passive classification shared by all states.
    metadata
        Optional string-keyed provenance shared by all states.

    Notes
    -----
    A material containing an unknown-temperature reference cannot also contain
    numeric-temperature states. Use :meth:`at` to resolve a run-specific
    mutable :class:`Material` before assigning it to an optical component.
    """

    name: str
    """Human-readable physical material name."""
    active: bool
    """Whether resolved conditions participate in population dynamics."""
    metadata: Mapping[str, Any]
    """String-keyed provenance shared by every temperature state."""

    def __init__(self, name: str, *, active: bool, metadata: Mapping[str, Any] | None = None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("material name must be a non-empty string")
        if not isinstance(active, (bool, np.bool_)):
            raise TypeError("material active must be a boolean")
        self.name = name
        self.active = bool(active)
        self.metadata = _metadata_copy(metadata)
        self._states: list[_MaterialState] = []

    @property
    def states(self):
        """Temperature-ordered immutable tuple of internal state values."""
        return tuple(self._states)

    def addState(
        self,
        *,
        temperature,
        refractiveIndex,
        fluorescenceLifetime=None,
        crossSections=None,
        bulkAttenuation=None,
        absorptionCoefficient=None,
        metadata=None,
    ):
        """Add one measured or modelled temperature state.

        Parameters
        ----------
        temperature
            Positive absolute temperature, or ``None`` with an explanatory
            ``metadata["temperature_status"]``.
        refractiveIndex
            Positive dimensionless optical index.
        fluorescenceLifetime
            Optional positive time quantity.
        crossSections
            Optional :class:`CrossSectionTable`.
        bulkAttenuation
            Optional non-negative inverse-length passive loss coefficient.
        absorptionCoefficient
            Alias for ``bulkAttenuation``; provide at most one of the two.
        metadata
            Optional state-specific provenance; it overrides same-named
            material metadata after selection.

        Returns
        -------
        _MaterialRecord
            ``self`` for chained state definitions.
        """
        if bulkAttenuation is not None and absorptionCoefficient is not None:
            raise TypeError("provide either bulkAttenuation or absorptionCoefficient, not both")
        state = _MaterialState(
            temperature=temperature,
            refractiveIndex=refractiveIndex,
            fluorescenceLifetime=fluorescenceLifetime,
            crossSections=crossSections,
            bulkAttenuation=(
                absorptionCoefficient
                if bulkAttenuation is None
                else bulkAttenuation
            ),
            metadata={} if metadata is None else metadata,
        )
        self._append_state(state)
        return self

    def _append_state(self, state: _MaterialState):
        if not isinstance(state, _MaterialState):
            raise TypeError("state must be an internal material state")
        if state.temperature is None:
            if self._states:
                raise ValueError("an unknown-temperature reference cannot be combined with other states")
        elif any(existing.temperature is None for existing in self._states):
            raise ValueError("a numeric temperature cannot be added beside an unknown-temperature reference")
        else:
            temperature_si = float(state.temperature.toValue(units.K))
            if any(
                np.isclose(
                    temperature_si,
                    float(existing.temperature.toValue(units.K)),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for existing in self._states
            ):
                raise ValueError(f"material already has a state at {temperature_si:g} K")
        self._states.append(state)
        self._states.sort(
            key=lambda value: -np.inf
            if value.temperature is None
            else float(value.temperature.toValue(units.K))
        )

    def at(
        self,
        *,
        temperature=None,
        activeIonDensity=Quantity(0.0, units.m**-3),
        spectralResolution=None,
        interpolation="exact",
        name=None,
        opticalAxis=None,
    ):
        """Resolve one material state and run-specific active-ion density.

        Parameters
        ----------
        temperature
            Required absolute temperature for numerically temperature-resolved
            materials. Omit it for an unknown-temperature reference.
        activeIonDensity
            Non-negative total active-ion number density; defaults to zero and
            does not alter the record's active/passive classification. A
            ``GainMedium`` requires a positive value for an active material.
        spectralResolution
            Optional integer target cross-section sample count. It must not be
            smaller than the selected table. Resampling occurs in the material
            layer before transport and is recorded in spectrum metadata.
        interpolation
            ``"exact"`` (default) requires a stored temperature.
            ``"linear"`` interpolates between two bracketing states, emits
            :class:`TemperatureInterpolationWarning`, and never extrapolates.
        name
            Optional run-specific condition label.
        opticalAxis
            Optional finite non-zero orientation vector, normalized on output.

        Returns
        -------
        Material
            Mutable physical properties ready for an optical component.
        """
        if interpolation not in {"exact", "linear"}:
            raise ValueError("temperature interpolation must be 'exact' or 'linear'")
        if not self._states:
            raise ValueError(f"material {self.name!r} contains no temperature states")
        if self._states[0].temperature is None:
            if temperature is not None:
                raise ValueError(
                    f"material {self.name!r} has no documented reference temperature; "
                    "a temperature-specific state must be added before selecting one"
                )
            state = self._states[0]
        else:
            temperature = requireQuantity(temperature, TEMPERATURE, "material temperature")
            state = self._state_at_temperature(temperature, interpolation)
        crossSections = state.crossSections
        if spectralResolution is not None:
            if crossSections is None:
                raise ValueError(
                    f"material {self.name!r} has no cross sections to resample"
                )
            crossSections = crossSections.resampleLinear(spectralResolution)
        metadata = dict(self.metadata)
        metadata.update(state.metadata)
        metadata["temperature_interpolation"] = interpolation
        return Material(
            materialName=self.name,
            temperature=state.temperature,
            refractiveIndex=state.refractiveIndex,
            fluorescenceLifetime=state.fluorescenceLifetime,
            crossSections=crossSections,
            active=self.active,
            bulkAttenuation=state.bulkAttenuation,
            activeIonDensity=activeIonDensity,
            name=name,
            opticalAxis=opticalAxis,
            metadata=metadata,
        )

    def _state_at_temperature(self, temperature, interpolation):
        target = float(temperature.toValue(units.K))
        known = np.asarray(
            [float(state.temperature.toValue(units.K)) for state in self._states],
            dtype=np.float64,
        )
        exact = np.flatnonzero(np.isclose(known, target, rtol=0.0, atol=1.0e-12))
        if exact.size:
            return self._states[int(exact[0])]
        if interpolation != "linear":
            available = ", ".join(f"{value:g} K" for value in known)
            raise KeyError(f"no {self.name} material state at {target:g} K; available: {available}")
        upper_index = int(np.searchsorted(known, target))
        if upper_index == 0 or upper_index == len(known):
            raise ValueError("temperature extrapolation is not supported")
        lower = self._states[upper_index - 1]
        upper = self._states[upper_index]
        warnings.warn(
            f"no {self.name} material state exists at {target:g} K; "
            f"interpolating between {known[upper_index - 1]:g} K and "
            f"{known[upper_index]:g} K",
            TemperatureInterpolationWarning,
            stacklevel=2,
        )
        weight = (target - known[upper_index - 1]) / (known[upper_index] - known[upper_index - 1])
        return _interpolate_state(lower, upper, temperature, weight)

def _default_key(name):
    result = "".join(character for character in name if character.isalnum())
    if not result or result[0].isdigit():
        result = "material_" + result
    return result


def _interpolate_optional_quantity(lower, upper, weight, name):
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError(f"cannot interpolate {name} when it is absent from one temperature state")
    return Quantity(
        np.asarray(lower.magnitude) * (1.0 - weight) + np.asarray(upper.toValue(lower.unit)) * weight,
        lower.unit,
    )


def _interpolate_cross_sections(lower, upper, weight):
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError("cannot interpolate cross sections absent from one temperature state")
    wavelength_unit = lower.wavelengths.unit
    wavelengths = np.unique(
        np.concatenate(
            (
                np.asarray(lower.wavelengths.magnitude),
                np.asarray(upper.wavelengths.toValue(wavelength_unit)),
            )
        )
    )
    lower_wavelengths = np.asarray(lower.wavelengths.toValue(wavelength_unit))
    upper_wavelengths = np.asarray(upper.wavelengths.toValue(wavelength_unit))
    absorption_unit = lower.absorption.unit
    emission_unit = lower.emission.unit
    lower_absorption = np.interp(wavelengths, lower_wavelengths, lower.absorption.magnitude)
    upper_absorption = np.interp(
        wavelengths, upper_wavelengths, upper.absorption.toValue(absorption_unit)
    )
    lower_emission = np.interp(wavelengths, lower_wavelengths, lower.emission.magnitude)
    upper_emission = np.interp(
        wavelengths, upper_wavelengths, upper.emission.toValue(emission_unit)
    )
    return CrossSectionTable(
        Quantity(wavelengths, wavelength_unit),
        Quantity(lower_absorption * (1.0 - weight) + upper_absorption * weight, absorption_unit),
        Quantity(lower_emission * (1.0 - weight) + upper_emission * weight, emission_unit),
        metadata={"temperature_interpolation": "linear"},
    )


def _interpolate_state(lower, upper, temperature, weight):
    metadata = dict(lower.metadata)
    metadata.update(upper.metadata)
    metadata["interpolated_between_K"] = [
        float(lower.temperature.toValue(units.K)),
        float(upper.temperature.toValue(units.K)),
    ]
    return _MaterialState(
        temperature=temperature,
        refractiveIndex=lower.refractiveIndex * (1.0 - weight) + upper.refractiveIndex * weight,
        fluorescenceLifetime=_interpolate_optional_quantity(
            lower.fluorescenceLifetime,
            upper.fluorescenceLifetime,
            weight,
            "fluorescenceLifetime",
        ),
        crossSections=_interpolate_cross_sections(lower.crossSections, upper.crossSections, weight),
        bulkAttenuation=_interpolate_optional_quantity(
            lower.bulkAttenuation, upper.bulkAttenuation, weight, "bulkAttenuation"
        ),
        metadata=metadata,
    )
