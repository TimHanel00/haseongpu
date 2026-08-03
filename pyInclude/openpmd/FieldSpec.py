from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hase_units import Quantity, Unit


@dataclass(frozen=True)
class FieldSpec:
    """Resolved openPMD field declaration used by transport encoders.

    Parameters
    ----------
    name
        Python-side field name.
    recordName
        HASE record suffix written below the openPMD ``meshes`` group.
    axes
        Ordered logical axes, for example ``("cell", "local_face")``.
    dtype
        NumPy-compatible storage dtype.
    shape
        Callable receiving a topology context and returning the expected shape.
    unit
        Unit symbol or :class:`Unit`. A Unit fills ``unitSI`` and
        ``unitDimension`` automatically.
    unitSI, unitDimension
        openPMD scale to SI and seven base-dimension exponents.
    dynamic
        Whether values may change after the first transport iteration.
    backendRequired
        Whether native parsing requires this field.
    userDefined
        Whether applications rather than HASE own the field.
    schemaRole
        Logical role such as ``"input"`` or ``"result"``.
    """
    name: str
    """Python-side field name used by schema-aware application code."""
    recordName: str
    """Exact openPMD record suffix written below ``meshes``."""
    axes: tuple[str, ...]
    """Ordered logical axes that define the primitive array layout."""
    dtype: object
    """NumPy-compatible scalar storage type."""
    shape: Callable[[object], tuple[int, ...]]
    """Callable resolving the primitive shape from a topology context."""
    unit: str = "1"
    """Human-readable physical unit symbol stored in metadata."""
    unitSI: float = 1.0
    """Multiplier converting stored magnitudes to SI."""
    unitDimension: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """Seven openPMD SI base-dimension exponents."""
    dynamic: bool = False
    """Whether values may change after the initial transport iteration."""
    backendRequired: bool = True
    """Whether native parsing requires the field to be present."""
    userDefined: bool = False
    """Whether application code, rather than HASE, owns the field values."""
    schemaRole: str = "input"
    """Logical role, normally ``"input"`` or ``"result"``."""

    def __post_init__(self):
        if isinstance(self.unit, Unit):
            object.__setattr__(self, "unitSI", self.unit.unitSI)
            object.__setattr__(self, "unitDimension", self.unit.unitDimension)
            object.__setattr__(self, "unit", self.unit.symbol)

    def expectedShape(self, context) -> tuple[int, ...]:
        """Resolve and normalize the declared shape for ``context``."""
        return tuple(int(size) for size in self.shape(context))

    @property
    def dtypeObject(self):
        """Storage dtype as :class:`numpy.dtype`."""
        return np.dtype(self.dtype)

    @property
    def entity(self) -> str:
        """Axes joined into the legacy entity label."""
        return "_".join(self.axes)

    @property
    def physicalUnit(self):
        """Resolved physical :class:`Unit` from the openPMD metadata."""
        return Unit.fromOpenPmd(self.unit, self.unitSI, self.unitDimension)

    def quantity(self, values):
        """Wrap stored magnitudes as a quantity in this field's unit."""
        return Quantity(values, self.physicalUnit)

    def storageValue(self, quantity):
        """Convert a compatible quantity to this field's stored magnitudes."""
        if not isinstance(quantity, Quantity):
            raise TypeError(f"{self.name} requires a unit-bearing Quantity")
        return quantity.toValue(self.physicalUnit)
