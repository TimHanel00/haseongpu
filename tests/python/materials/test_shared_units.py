# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util

import HASEonGPU
import hase_units

from material_library import CrossSectionTable


def test_hase_and_material_library_share_the_independent_unit_types():
    table = CrossSectionTable.monochromatic(
        wavelength=1030 * hase_units.units.nm,
        absorption=1.0e-21 * hase_units.units.cm**2,
        emission=2.0e-20 * hase_units.units.cm**2,
    )

    assert HASEonGPU.Unit is hase_units.Unit
    assert HASEonGPU.Quantity is hase_units.Quantity
    assert HASEonGPU.units is hase_units.units
    assert isinstance(table.wavelengths, hase_units.Quantity)


def test_material_library_has_no_units_compatibility_module():
    assert importlib.util.find_spec("material_library.units") is None
