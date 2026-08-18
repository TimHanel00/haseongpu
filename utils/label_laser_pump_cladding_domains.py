#!/usr/bin/env python3
# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Label the gain core and radial cladding cells of the pump example mesh."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
from pathlib import Path

import numpy as np


converterPath = Path(__file__).resolve().with_name("convert_vtk_topology.py")
converterSpec = importlib.util.spec_from_file_location(
    "_hase_convert_vtk_topology",
    converterPath,
)
if converterSpec is None or converterSpec.loader is None:
    raise RuntimeError(f"Could not load VTK conversion helpers from '{converterPath}'")
converter = importlib.util.module_from_spec(converterSpec)
converterSpec.loader.exec_module(converter)


GAIN_DOMAIN = np.int32(1)
CLADDING_DOMAIN = np.int32(2)


def _boundaryTriangleMask(triangles):
    """Select cross-section triangles containing an exposed mesh edge."""
    edges = (
        (triangle[index], triangle[(index + 1) % 3])
        for triangle in np.asarray(triangles, dtype=np.uint32)
        for index in range(3)
    )
    counts = Counter(tuple(sorted((int(left), int(right)))) for left, right in edges)
    return np.asarray(
        [
            any(
                counts[tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))]
                == 1
                for index in range(3)
            )
            for triangle in triangles
        ],
        dtype=bool,
    )


def laserPumpCladdingCellDomains(points, cells, numberOfLevels):
    """Return gain/cladding ids for an axially extruded three-Tet wedge mesh.

    The cladding is one cross-section cell thick. A base triangle belongs to
    the cladding when one of its edges is exposed at the radial boundary. Its
    classification is copied through every axial layer and to all three Tet4
    children of the corresponding wedge.
    """
    cells = np.asarray(cells, dtype=np.uint32)
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.shape[0] % 3:
        raise ValueError("laserPumpCladding partition requires three Tet4 cells per wedge")
    if numberOfLevels < 2:
        raise ValueError("laserPumpCladding partition requires at least two axial levels")
    wedgeGroups = cells.reshape((-1, 3, 4))
    if wedgeGroups.shape[0] % (numberOfLevels - 1):
        raise ValueError("wedge count is not divisible by the number of axial intervals")
    wedgesPerLayer = wedgeGroups.shape[0] // (numberOfLevels - 1)
    baseWedges = np.asarray(
        [
            converter._wedge_from_tetrahedra(points, group)
            for group in wedgeGroups[:wedgesPerLayer]
        ],
        dtype=np.uint32,
    )
    baseCladding = _boundaryTriangleMask(baseWedges[:, :3])
    wedgeCladding = np.broadcast_to(
        baseCladding,
        (numberOfLevels - 1, wedgesPerLayer),
    ).reshape(-1)
    return np.where(np.repeat(wedgeCladding, 3), CLADDING_DOMAIN, GAIN_DOMAIN)


def labelLaserPumpCladdingDomains(inputPath, outputPath):
    """Write the example topology with deterministic gain/cladding labels."""
    points, cells, cellTypes, pointData, cellData, fields = converter._parseVtk(inputPath)
    levels = int(np.asarray(fields["structuredNumberOfLevels"]).reshape(-1)[0])
    fields = dict(fields)
    fields["cellDomains"] = laserPumpCladdingCellDomains(points, cells, levels)
    return converter._write_ascii_vtk(
        outputPath,
        title="HASEonGPU laserPumpCladding Tet4 topology",
        points=np.asarray(points, dtype=np.float64),
        cells=np.asarray(cells, dtype=np.uint32),
        cell_types=np.asarray(cellTypes, dtype=np.uint32),
        fields=fields,
        point_data=pointData,
        cell_data=cellData,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(labelLaserPumpCladdingDomains(args.input, args.output))


if __name__ == "__main__":
    main()
