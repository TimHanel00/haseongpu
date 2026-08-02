# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from types import SimpleNamespace

from example import _source_tree_import


def test_example_helper_accepts_installed_material_mesh_api(monkeypatch):
    installed = SimpleNamespace(UnstructuredMesh=object(), MaterialDefinition=object())
    monkeypatch.setitem(sys.modules, "HASEonGPU", installed)

    def unexpected_fallback():
        raise AssertionError("compatible installed frontend was rejected")

    monkeypatch.setattr(_source_tree_import, "_clear_hase_modules", unexpected_fallback)

    _source_tree_import.ensure_hase_importable()
    assert sys.modules["HASEonGPU"] is installed
