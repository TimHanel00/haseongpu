#!/usr/bin/env python3
# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convert a legacy four-file optical material table to the HDF5 database."""

from __future__ import annotations

import argparse

from . import CrossSectionTable, Material, MaterialLibrary
from hase_units import units


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory")
    parser.add_argument("output_file")
    parser.add_argument("--key", required=True, help="Python identifier used by MaterialLibrary")
    parser.add_argument("--name", required=True)
    parser.add_argument("--refractive-index", type=float, required=True)
    parser.add_argument("--fluorescence-lifetime-seconds", type=float, required=True)
    temperature = parser.add_mutually_exclusive_group(required=True)
    temperature.add_argument("--temperature-kelvin", type=float)
    temperature.add_argument(
        "--temperature-unknown-reason",
        help="explicit explanation when the source does not document a temperature",
    )
    parser.add_argument("--source", help="citation, DOI, URL, or other provenance")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    state_metadata = {}
    if args.source:
        state_metadata["source"] = args.source
    if args.temperature_kelvin is None:
        state_metadata["temperature_status"] = args.temperature_unknown_reason
        temperature = None
    else:
        temperature = args.temperature_kelvin * units.K
    spectra = CrossSectionTable.fromTextDirectory(
        args.input_directory,
        metadata={"source": args.source} if args.source else None,
    )
    material = Material(args.name).addState(
        temperature=temperature,
        refractiveIndex=args.refractive_index,
        fluorescenceLifetime=args.fluorescence_lifetime_seconds * units.s,
        crossSections=spectra,
        metadata=state_metadata,
    )
    library = MaterialLibrary()
    library.register(args.key, material)
    library.toHdf5(args.output_file, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
