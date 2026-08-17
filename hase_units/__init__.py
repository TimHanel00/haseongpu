# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared unit-bearing physical values with native openPMD metadata."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


__all__ = [
    "AMOUNT",
    "CURRENT",
    "DIMENSIONLESS",
    "LENGTH",
    "LUMINOUS_INTENSITY",
    "MASS",
    "POWER",
    "Quantity",
    "TEMPERATURE",
    "TIME",
    "Unit",
    "requireQuantity",
    "units",
]


DIMENSIONLESS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LENGTH = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TIME = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
MASS = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
CURRENT = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
TEMPERATURE = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
AMOUNT = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
LUMINOUS_INTENSITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
POWER = (2.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0)


def _combine_dimensions(left, right, operation):
    return tuple(operation(float(a), float(b)) for a, b in zip(left, right))


def _format_product(left, right, operator):
    if left == "1" and operator == "*":
        return right
    if right == "1":
        return left
    return f"{left} {operator} {right}"


@dataclass(frozen=True)
class Unit:
    """Linear physical unit with native openPMD metadata.

    Parameters
    ----------
    symbol
        Human-readable unit expression, such as ``"nm"`` or ``"cm^-3"``.
    unitSI
        Positive scale converting a magnitude in this unit to SI.
    unitDimension
        Seven SI base-dimension exponents in openPMD order: length, mass,
        time, electric current, temperature, amount, luminous intensity.

    Notes
    -----
    Multiplication, division, and powers create derived units. Multiplying a
    numeric value by a unit creates a :class:`Quantity`.
    """

    symbol: str
    """Human-readable unit expression used in reprs and generated metadata."""
    unitSI: float
    """Positive multiplier that converts a magnitude in this unit to SI."""
    unitDimension: tuple[float, float, float, float, float, float, float]
    """Seven SI base-dimension exponents in openPMD ordering."""

    __array_priority__ = 10_000

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("unit symbol must be a non-empty string")
        if not np.isfinite(self.unitSI) or self.unitSI <= 0.0:
            raise ValueError("unitSI must be finite and positive")
        dimensions = tuple(float(value) for value in self.unitDimension)
        if len(dimensions) != 7 or np.any(~np.isfinite(dimensions)):
            raise ValueError("unitDimension must contain seven finite SI exponents")
        object.__setattr__(self, "unitDimension", dimensions)

    @classmethod
    def fromOpenPmd(cls, symbol, unitSI, unitDimension):
        """Construct a unit from openPMD metadata.

        Parameters
        ----------
        symbol
            Human-readable ``unit`` metadata.
        unitSI
            Positive ``unitSI`` conversion factor.
        unitDimension
            Seven-component ``unitDimension`` sequence.
        """
        return cls(str(symbol), float(unitSI), tuple(unitDimension))

    @property
    def openPmdMetadata(self):
        """Return this unit as an openPMD metadata mapping."""
        return {
            "unit": self.symbol,
            "unitSI": self.unitSI,
            "unitDimension": self.unitDimension,
        }

    def isCompatible(self, other):
        """Return whether ``other`` has the same physical dimensions.

        Parameters
        ----------
        other
            Candidate :class:`Unit`.
        """
        return isinstance(other, Unit) and self.unitDimension == other.unitDimension

    def __rmul__(self, magnitude):
        return Quantity(magnitude, self)

    def __rtruediv__(self, magnitude):
        return Quantity(magnitude, self**-1)

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit(
                _format_product(self.symbol, other.symbol, "*"),
                self.unitSI * other.unitSI,
                _combine_dimensions(self.unitDimension, other.unitDimension, lambda a, b: a + b),
            )
        return Quantity(other, self)

    def __truediv__(self, other):
        if not isinstance(other, Unit):
            return Quantity(1.0 / other, self)
        return Unit(
            _format_product(self.symbol, other.symbol, "/"),
            self.unitSI / other.unitSI,
            _combine_dimensions(self.unitDimension, other.unitDimension, lambda a, b: a - b),
        )

    def __pow__(self, exponent):
        exponent = float(exponent)
        shown = int(exponent) if exponent.is_integer() else exponent
        symbol = self.symbol if exponent == 1.0 else f"{self.symbol}^{shown}"
        return Unit(
            symbol,
            self.unitSI**exponent,
            tuple(value * exponent for value in self.unitDimension),
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        if ufunc is np.multiply:
            magnitude = inputs[1] if inputs[0] is self else inputs[0]
            return Quantity(magnitude, self)
        if ufunc in (np.true_divide, np.divide) and inputs[1] is self:
            return Quantity(inputs[0], self**-1)
        return NotImplemented


@dataclass(frozen=True)
class Quantity:
    """Numeric scalar or array together with a physical :class:`Unit`.

    Parameters
    ----------
    magnitude
        Numeric scalar or array expressed in ``unit``. Array values are copied
        and made read-only.
    unit
        Physical unit associated with the magnitude.

    Notes
    -----
    Arithmetic preserves or combines units. Addition and subtraction require
    compatible dimensions and convert the right operand to the left unit.
    ``numpy.asarray(quantity)`` exposes the stored magnitude, not its SI value.
    """

    magnitude: object
    """Numeric scalar or immutable array expressed in :attr:`unit`."""
    unit: Unit
    """Physical unit carried by :attr:`magnitude`."""

    __array_priority__ = 10_000

    def __post_init__(self):
        if not isinstance(self.unit, Unit):
            raise TypeError("Quantity.unit must be Unit")
        values = np.asarray(self.magnitude)
        if values.dtype.kind in {"O", "S", "U", "V"}:
            raise TypeError("quantity magnitude must be numeric")
        if values.ndim == 0:
            value = values.item()
        else:
            value = values.copy()
            value.flags.writeable = False
        object.__setattr__(self, "magnitude", value)

    @property
    def value(self):
        """Stored magnitude in :attr:`unit`; alias for ``magnitude``."""
        return self.magnitude

    @property
    def unitSI(self):
        """Scale converting the stored magnitude to SI."""
        return self.unit.unitSI

    @property
    def unitDimension(self):
        """Seven SI base-dimension exponents of this quantity."""
        return self.unit.unitDimension

    @property
    def openPmdMetadata(self):
        """Return the associated unit's openPMD metadata mapping."""
        return self.unit.openPmdMetadata

    @property
    def siValue(self):
        """Magnitude converted to SI base units as a NumPy value."""
        return np.asarray(self.magnitude) * self.unit.unitSI

    @property
    def shape(self):
        """Shape of the stored magnitude."""
        return np.shape(self.magnitude)

    @property
    def size(self):
        """Number of scalar values in the stored magnitude."""
        return np.size(self.magnitude)

    def to(self, unit):
        """Return an equivalent quantity expressed in a compatible unit.

        Parameters
        ----------
        unit
            Target :class:`Unit` with identical physical dimensions.
        """
        return Quantity(self.toValue(unit), unit)

    def toValue(self, unit):
        """Return only the magnitude expressed in a compatible target unit.

        Parameters
        ----------
        unit
            Target :class:`Unit` with identical physical dimensions.
        """
        if not self.unit.isCompatible(unit):
            raise ValueError(
                f"cannot convert {self.unit.symbol} to dimensionally incompatible {unit.symbol}"
            )
        return np.asarray(self.magnitude) * (self.unit.unitSI / unit.unitSI)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.magnitude, dtype=dtype)

    def __float__(self):
        values = np.asarray(self.magnitude)
        if values.ndim != 0:
            raise TypeError("only scalar quantities can be converted to float")
        return float(values)

    def __repr__(self):
        return f"Quantity({self.magnitude!r}, {self.unit.symbol})"

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(np.asarray(self.magnitude) * np.asarray(other.magnitude), self.unit * other.unit)
        if isinstance(other, Unit):
            return Quantity(self.magnitude, self.unit * other)
        return Quantity(np.asarray(self.magnitude) * other, self.unit)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(np.asarray(self.magnitude) / np.asarray(other.magnitude), self.unit / other.unit)
        if isinstance(other, Unit):
            return Quantity(self.magnitude, self.unit / other)
        return Quantity(np.asarray(self.magnitude) / other, self.unit)

    def __rtruediv__(self, other):
        return Quantity(other / np.asarray(self.magnitude), self.unit**-1)

    def __add__(self, other):
        if not isinstance(other, Quantity):
            return NotImplemented
        return Quantity(np.asarray(self.magnitude) + other.toValue(self.unit), self.unit)

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            return NotImplemented
        return Quantity(np.asarray(self.magnitude) - other.toValue(self.unit), self.unit)


def requireQuantity(value, dimension, name, *, allowNone=False):
    """Validate that a public input is a quantity of the expected dimension.

    Parameters
    ----------
    value
        Candidate :class:`Quantity`.
    dimension
        Seven-component SI dimension tuple to require.
    name
        Parameter name used in validation errors.
    allowNone
        Return ``None`` unchanged when true.

    Returns
    -------
    Quantity or None
        The original validated object.
    """
    if value is None and allowNone:
        return None
    if not isinstance(value, Quantity):
        raise TypeError(f"{name} must be a unit-bearing Quantity")
    expected = tuple(float(item) for item in dimension)
    if value.unitDimension != expected:
        raise ValueError(
            f"{name} has unit {value.unit.symbol} with dimension {value.unitDimension}; "
            f"expected {expected}"
        )
    return value


#: Catalogue of predefined units used by public HASEonGPU inputs. Derived units
#: are formed with multiplication, division, and powers, for example
#: ``2.7e20 / units.cm**3``.
units = SimpleNamespace(
    one=Unit("1", 1.0, DIMENSIONLESS),
    m=Unit("m", 1.0, LENGTH),
    cm=Unit("cm", 1.0e-2, LENGTH),
    mm=Unit("mm", 1.0e-3, LENGTH),
    um=Unit("um", 1.0e-6, LENGTH),
    nm=Unit("nm", 1.0e-9, LENGTH),
    s=Unit("s", 1.0, TIME),
    ms=Unit("ms", 1.0e-3, TIME),
    us=Unit("us", 1.0e-6, TIME),
    kg=Unit("kg", 1.0, MASS),
    g=Unit("g", 1.0e-3, MASS),
    A=Unit("A", 1.0, CURRENT),
    K=Unit("K", 1.0, TEMPERATURE),
    mK=Unit("mK", 1.0e-3, TEMPERATURE),
    mol=Unit("mol", 1.0, AMOUNT),
    cd=Unit("cd", 1.0, LUMINOUS_INTENSITY),
    W=Unit("W", 1.0, POWER),
    kW=Unit("kW", 1.0e3, POWER),
)
