# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


def _build_binding_dirs():
    source_root = Path(__file__).resolve().parents[1]
    candidates = [source_root / "build" / "python" / "HASEonGPU_Bindings"]
    candidates.extend(
        sorted(
            source_root.glob("build/cp*/python/HASEonGPU_Bindings"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.is_dir():
            yield candidate


for _build_bindings in _build_binding_dirs():
    _build_path = str(_build_bindings)
    if _build_path not in __path__:
        __path__.insert(0, _build_path)
try:
    del _build_bindings
    del _build_path
except NameError:
    pass

from .HASEonGPU import *
