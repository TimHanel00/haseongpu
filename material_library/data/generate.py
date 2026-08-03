# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate the bundled HDF5 material database from repository source data."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from material_library import (
    CrossSectionTable,
    LegacyMaterialTextWarning,
    Material,
    MaterialLibrary,
)
from hase_units import units


DATA_DIRECTORY = Path(__file__).resolve().parent
DATABASE_PATH = DATA_DIRECTORY / "materials.h5"
YB_YAG_ROOM_TEMPERATURE = 293.15 * units.K


def buildMaterialLibrary():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyMaterialTextWarning)
        crossSections = CrossSectionTable.fromTextDirectory(
            DATA_DIRECTORY / "legacy_yb_yag",
            metadata={
                "source": "legacy HASEonGPU laserPumpCladding dataset",
                "source_format": "legacy-four-file-text",
            },
        )
    yb_yag = Material("Yb:YAG").addState(
        temperature=YB_YAG_ROOM_TEMPERATURE,
        refractiveIndex=1.83,
        fluorescenceLifetime=0.941 * units.ms,
        crossSections=cross_sections,
        metadata={
            "source": "legacy HASEonGPU laserPumpCladding dataset",
            "temperature_context": "cross sections recorded at room temperature",
        },
    )
    library = MaterialLibrary()
    library.register("YbYAG", yb_yag)
    return library


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DATABASE_PATH)
    args = parser.parse_args(argv)
    buildMaterialLibrary().toHdf5(args.output, overwrite=True)


if __name__ == "__main__":
    main()
