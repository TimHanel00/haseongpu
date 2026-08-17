# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Repository-provided material database."""

from pathlib import Path

from .library import MaterialLibrary


DATABASE_PATH = Path(__file__).resolve().parent / "data" / "materials.h5"


def loadBuiltinMaterials():
    """Load and validate the versioned material database shipped with HASE.

    Returns a new :class:`MaterialLibrary` instance, so registering additional
    application materials does not mutate another caller's library.
    """
    return MaterialLibrary.fromHdf5(DATABASE_PATH)
