# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standalone, unit-aware, temperature-resolved material database."""

from .builtins import DATABASE_PATH, loadBuiltinMaterials
from .library import MaterialLibrary
from .model import (
    CrossSectionTable,
    LegacyMaterialTextWarning,
    Material,
    MaterialCondition,
    MaterialState,
    TemperatureInterpolationWarning,
)


__all__ = [
    "CrossSectionTable",
    "DATABASE_PATH",
    "LegacyMaterialTextWarning",
    "loadBuiltinMaterials",
    "Material",
    "MaterialCondition",
    "MaterialLibrary",
    "MaterialState",
    "TemperatureInterpolationWarning",
]
