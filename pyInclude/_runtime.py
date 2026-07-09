# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runtime artifact discovery for Python frontends using external HASE builds."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def _configured_runtime_dir():
    try:
        native_config = importlib.import_module("pyInclude._native_config")
    except ImportError:
        return ""
    return str(getattr(native_config, "HASE_RUNTIME_DIR", "") or "")


def _path_entries(value):
    if not value:
        return
    for entry in str(value).split(os.pathsep):
        if entry:
            yield Path(entry)


def _unique(paths):
    seen = set()
    for path in paths:
        normalized = str(path.expanduser().resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        yield path.expanduser()


def runtime_roots():
    """Return configured native runtime roots before packaged fallback paths."""
    env_value = os.environ.get("HASE_RUNTIME_DIR", "")
    yield from _unique([*_path_entries(env_value), *_path_entries(_configured_runtime_dir())])


def native_dirs_from_root(root):
    """Yield plausible native artifact directories below a runtime root."""
    root = Path(root).expanduser()
    yield root
    yield root / "bin"
    yield root / "lib"
    yield root / "pyInclude" / "_native"


def runtime_native_dirs():
    """Return native artifact directories from all configured runtime roots."""
    for root in runtime_roots():
        yield from _unique(native_dirs_from_root(root))


def runtime_executable_candidates(names):
    """Return executable candidates below configured runtime roots."""
    for root in runtime_roots():
        root = Path(root).expanduser()
        for native_dir in _unique((root / "bin", root, root / "pyInclude" / "_native", root / "lib")):
            for name in names:
                yield native_dir / name


def runtime_library_candidates(names):
    """Return shared-library candidates below configured runtime roots."""
    for root in runtime_roots():
        root = Path(root).expanduser()
        for native_dir in _unique((root / "lib", root / "bin", root, root / "pyInclude" / "_native")):
            for name in names:
                yield native_dir / name
