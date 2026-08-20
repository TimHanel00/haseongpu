# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate the bundled HDF5 material database from repository source data."""

from __future__ import annotations

import argparse
import hashlib
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
LEGACY_YB_YAG_FILES = (
    "lambda_a.txt",
    "sigma_a.txt",
    "lambda_e.txt",
    "sigma_e.txt",
)


def _sourceMetadata():
    sourceDirectory = DATA_DIRECTORY / "legacy_yb_yag"
    return {
        "source": "legacy HASEonGPU laserPumpCladding dataset",
        "source_format": "legacy-four-file-text",
        "generator": "python3 -m material_library.data.generate",
        "generator_revision": "2",
        "input_sha256": {
            name: hashlib.sha256((sourceDirectory / name).read_bytes()).hexdigest()
            for name in LEGACY_YB_YAG_FILES
        },
        "backend": "not applicable",
        "seed": "not applicable",
    }


def buildMaterialLibrary():
    sourceMetadata = _sourceMetadata()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyMaterialTextWarning)
        crossSections = CrossSectionTable.fromTextDirectory(
            DATA_DIRECTORY / "legacy_yb_yag",
            metadata=sourceMetadata,
        )
    ybYag = Material(
        materialName="Yb:YAG",
        temperature=YB_YAG_ROOM_TEMPERATURE,
        refractiveIndex=1.83,
        fluorescenceLifetime=0.941 * units.ms,
        crossSections=crossSections,
        active=True,
        metadata=sourceMetadata
        | {
            "temperature_context": "cross sections recorded at room temperature",
        },
    )
    library = MaterialLibrary()
    library.register("YbYAG", ybYag)
    library.register(
        "CladdingGlass",
        Material(
            materialName="absorbing cladding glass",
            temperature=YB_YAG_ROOM_TEMPERATURE,
            refractiveIndex=1.45,
            fluorescenceLifetime=None,
            crossSections=None,
            active=False,
            bulkAttenuation=5.5 / units.cm,
            metadata={
                "source": "HASEonGPU laserPumpCladding reference condition",
                "generator": "python3 -m material_library.data.generate",
                "generator_revision": "3",
                "temperature_context": "nominal room-temperature example value",
            },
        ),
    )
    return library


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DATABASE_PATH)
    args = parser.parse_args(argv)
    buildMaterialLibrary().toHdf5(args.output, overwrite=True)


if __name__ == "__main__":
    main()
