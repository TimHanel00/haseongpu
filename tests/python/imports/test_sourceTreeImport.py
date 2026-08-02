# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

import ast
import sys
from pathlib import Path
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


def test_public_frontend_examples_do_not_depend_on_private_pyinclude_modules():
    example_root = Path(__file__).resolve().parents[3] / "example"

    for filename in ("minimalExampleNewInterface.py", "gmshMinimalExample.py"):
        tree = ast.parse((example_root / filename).read_text(encoding="utf-8"), filename=filename)
        private_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pyInclude"))
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name.startswith("pyInclude") for alias in node.names)
                )
            )
        ]
        assert private_imports == [], f"{filename} imports private pyInclude modules"
