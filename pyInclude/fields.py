# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generic field storage metadata independent of transport serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BackendFlatArray:
    values: object


def backendFlat(values):
    """Mark a one-dimensional array as using canonical backend-flat order."""
    return BackendFlatArray(values)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    recordName: str
    axes: tuple[str, ...]
    dtype: object
    shape: Callable[[object], tuple[int, ...]]
    unit: str = "1"
    unitSI: float = 1.0
    unitDimension: tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    dynamic: bool = False
    backendRequired: bool = True
    userDefined: bool = False

    def expectedShape(self, context) -> tuple[int, ...]:
        return tuple(int(size) for size in self.shape(context))

    @property
    def dtypeObject(self):
        return np.dtype(self.dtype)

    @property
    def entity(self) -> str:
        return "_".join(self.axes)
