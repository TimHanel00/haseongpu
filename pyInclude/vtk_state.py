# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""ASCII VTK writer for public Tet4 simulation states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

def _field_label(field):
    """Create a stable filename label for one or more VTK fields."""
    if isinstance(field, Mapping):
        names = field.keys()
    elif isinstance(field, Sequence) and not isinstance(field, (str, bytes)):
        names = field
    else:
        names = (field,)
    return "_".join(str(name) for name in names)


def _vtk_filename(file_name, state=None, field="phiAse"):
    """Resolve ``{step}``, ``{time}``, and ``{field}`` filename placeholders."""
    text = str(file_name)
    if state is not None:
        text = text.format(
            step=getattr(state, "step", 0),
            time=getattr(state, "time", 0.0),
            field=_field_label(field),
        )
    if not text.endswith(".vtk"):
        text += ".vtk"
    return Path(text)


def _as_named_array(data, field, scalar_name):
    """Select a named scalar array from a state object or raw array."""
    if isinstance(data, Mapping) and field in data:
        values = data[field]
        name = scalar_name or field
    elif hasattr(data, field):
        values = getattr(data, field)
        name = scalar_name or field
    else:
        values = data
        name = scalar_name or "scalars"
    if values is None:
        raise ValueError(f"no data available for VTK field '{field}'")
    return name, np.asarray(values, dtype=np.float64)


def _scalar_name_for_field(scalar_name, field, index):
    if scalar_name is None:
        return None
    if isinstance(scalar_name, Mapping):
        return scalar_name.get(field)
    if isinstance(scalar_name, Sequence) and not isinstance(scalar_name, (str, bytes)):
        return scalar_name[index]
    return scalar_name


def _as_named_arrays(data, field, scalar_name=None, fields=None):
    """Resolve one or more named scalar arrays from object attributes or mappings."""
    if fields is not None:
        if not isinstance(fields, Mapping):
            raise TypeError("fields must be a mapping of scalar names to arrays or attribute names")
        arrays = []
        for name, source in fields.items():
            if isinstance(source, str) and hasattr(data, source):
                values = getattr(data, source)
            elif isinstance(data, Mapping) and isinstance(source, str) and source in data:
                values = data[source]
            else:
                values = source
            if values is None:
                raise ValueError(f"no data available for VTK field '{name}'")
            arrays.append((str(name), np.asarray(values, dtype=np.float64)))
        return arrays

    if isinstance(field, Mapping):
        arrays = []
        for index, (name, source) in enumerate(field.items()):
            alias = _scalar_name_for_field(scalar_name, name, index) or name
            if isinstance(source, str):
                _, values = _as_named_array(data, source, alias)
            else:
                values = np.asarray(source, dtype=np.float64)
            arrays.append((str(alias), values))
        return arrays

    if isinstance(field, Sequence) and not isinstance(field, (str, bytes)):
        return [
            _as_named_array(data, item, _scalar_name_for_field(scalar_name, item, index))
            for index, item in enumerate(field)
        ]

    return [_as_named_array(data, field, scalar_name)]


def _write_ascii_tet4(file_name, arrays, topology):
    """Write point- and cell-shaped arrays on a Tet4 topology."""
    points = np.asarray(topology.points, dtype=np.float64)
    cells = np.asarray(topology.cellPointIndices, dtype=np.uint32)
    point_count = points.shape[0]
    cell_count = cells.shape[0]
    point_fields = [
        (name, values)
        for name, values in arrays
        if np.asarray(values).size == point_count
    ]
    cell_fields = [
        (name, values)
        for name, values in arrays
        if np.asarray(values).size == cell_count
    ]

    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 2.0\n")
        handle.write("HASEonGPU Tet4 output\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {point_count} double\n")
        for x, y, z in points:
            handle.write(f"{x:.17g} {y:.17g} {z:.17g}\n")
        handle.write(f"CELLS {cell_count} {cell_count * 5}\n")
        for cell in cells:
            handle.write("4 " + " ".join(str(int(vertex)) for vertex in cell) + "\n")
        handle.write(f"CELL_TYPES {cell_count}\n")
        handle.write("10\n" * cell_count)
        for label, count, fields in (
            ("POINT_DATA", point_count, point_fields),
            ("CELL_DATA", cell_count, cell_fields),
        ):
            if not fields:
                continue
            handle.write(f"{label} {count}\n")
            for name, values in fields:
                handle.write(f"SCALARS {name} double 1\n")
                handle.write("LOOKUP_TABLE default\n")
                for value in np.asarray(values).reshape(-1, order="F"):
                    handle.write(f"{float(value):.17g}\n")
    return path


def writeVtkState(fileName, state, *, fields=None, field="phiAse"):
    """Write scalar arrays from a time-step state to an ASCII Tet4 VTK file.

    Parameters
    ----------
    fileName
        Destination path. ``.vtk`` is appended when absent. The string may use
        ``{step}``, ``{time}``, and ``{field}`` placeholders.
    state
        :class:`TimeStepState` containing ``mesh`` and the requested arrays.
    fields
        Optional mapping from VTK scalar names to arrays or state-attribute
        names. It takes precedence over ``field`` and writes several arrays in
        one file.
    field
        One state-attribute name, a sequence of names, or a mapping of output
        names to attributes/arrays. Defaults to ``"phiAse"``.

    Returns
    -------
    pathlib.Path
        Written file path.

    Notes
    -----
    An array whose size equals the mesh point count becomes ``POINT_DATA``;
    one matching the Tet4 count becomes ``CELL_DATA``. Other sizes are not
    written. This legacy ASCII writer supports scalar fields only.
    """
    topology = getattr(state, "mesh", None)
    if topology is None:
        raise TypeError("writeVtkState requires a state with mesh")
    arrays = _as_named_arrays(state, field, fields=fields)
    path = _vtk_filename(fileName, state, field)
    if not hasattr(topology, "cellPointIndices"):
        raise TypeError("writeVtkState requires Tet4 cell connectivity")
    return _write_ascii_tet4(path, arrays, topology)
