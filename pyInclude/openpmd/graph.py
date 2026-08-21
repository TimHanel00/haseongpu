"""Generic openPMD projection of a transport-neutral frontend graph."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from hase_transport import TransportGraph
from hase_units import DIMENSIONLESS, Quantity


TRANSPORT_VERSION = "1.1"
_RECORD_PREFIX = "hase__"
_ATTRIBUTE_PREFIX = "hase__attribute__"
_REFERENCE_PREFIX = "hase__reference__"


def encodePath(path: str) -> str:
    """Encode a logical primitive path as a provider-safe openPMD key."""
    return path.replace("%", "%25").replace("/", "%2F")


def decodePath(path: str) -> str:
    return path.replace("%2F", "/").replace("%25", "%")


def recordName(path: str) -> str:
    return _RECORD_PREFIX + encodePath(path)


def attributeName(path: str) -> str:
    return _ATTRIBUTE_PREFIX + encodePath(path)


def referenceName(path: str) -> str:
    return _REFERENCE_PREFIX + encodePath(path)


@dataclass(frozen=True)
class _NumericValue:
    values: np.ndarray
    unit: str
    unitSI: float
    unitDimension: tuple[float, ...]


def _numeric(value) -> _NumericValue:
    if isinstance(value, Quantity):
        return _NumericValue(
            np.asarray(value.magnitude),
            value.unit.symbol,
            float(value.unitSI),
            tuple(value.unitDimension),
        )
    values = np.asarray(value)
    if values.dtype.kind == "b":
        values = values.astype(np.uint8)
    if values.dtype.kind not in "biufc":
        raise TypeError(f"transport numeric field has unsupported dtype {values.dtype}")
    return _NumericValue(values, "1", 1.0, DIMENSIONLESS)


def _setRecordMetadata(record, *, path, typeName, spec, numeric, shape):
    record.set_attribute("haseTransportVersion", TRANSPORT_VERSION)
    record.set_attribute("hasePath", path)
    record.set_attribute("haseOwnerType", typeName)
    record.set_attribute("haseFieldName", spec.name)
    record.set_attribute("haseAxes", json.dumps(list(spec.axes), separators=(",", ":")))
    record.set_attribute(
        "haseShape",
        json.dumps([int(extent) for extent in shape], separators=(",", ":")),
    )
    record.set_attribute("haseDynamic", bool(spec.dynamic))
    record.set_attribute("haseEncoding", spec.encoding)
    record.set_attribute("haseUnit", numeric.unit)


def _unitDimension(io, dimensions):
    labels = (
        io.Unit_Dimension.L,
        io.Unit_Dimension.M,
        io.Unit_Dimension.T,
        io.Unit_Dimension.I,
        io.Unit_Dimension.theta,
        io.Unit_Dimension.N,
        io.Unit_Dimension.J,
    )
    return {
        label: float(exponent)
        for label, exponent in zip(labels, dimensions)
        if float(exponent) != 0.0
    }


def _writeNumeric(iteration, io, *, path, typeName, spec, value):
    numeric = _numeric(value)
    # openPMD keeps Python buffers alive until the collective flush.  Own the
    # memory here as well: derived topology arrays may be backed by Numba-owned
    # allocations that the Python binding cannot safely defer.
    source = np.asarray(numeric.values)
    # Numba may return an equivalent but non-canonical NumPy dtype object;
    # openPMD's Python binding dispatches using the canonical dtype identity.
    canonical_dtype = np.dtype(source.dtype.str)
    values = np.frombuffer(source.tobytes(order="C"), dtype=canonical_dtype).copy()
    record = iteration.meshes[recordName(path)]
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", "haseTransportGraph=1")
    record.set_attribute("dataOrder", "C")
    record.axis_labels = ["flatIndex"]
    record.grid_spacing = [1.0]
    record.grid_global_offset = [0.0]
    record.grid_unit_SI = 1.0
    record.unit_dimension = _unitDimension(io, numeric.unitDimension)
    component = record[io.Mesh_Record_Component.SCALAR]
    component.unit_SI = numeric.unitSI
    component.position = [0.0]
    component.reset_dataset(io.Dataset(values.dtype, values.shape))
    try:
        component.store_chunk(values)
    except Exception as exc:
        raise RuntimeError(
            f"failed to store transport field {path!r} with dtype {values.dtype} "
            f"and shape {values.shape}"
        ) from exc
    _setRecordMetadata(
        record,
        path=path,
        typeName=typeName,
        spec=spec,
        numeric=numeric,
        shape=numeric.values.shape,
    )
    return values


def _writeRagged(iteration, io, *, path, typeName, spec, value):
    arrays = [np.asarray(item).reshape(-1) for item in value]
    dtype = np.result_type(*(array.dtype for array in arrays)) if arrays else np.dtype(np.float64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.uint64)
    for index, array in enumerate(arrays):
        offsets[index + 1] = offsets[index] + array.size
    values = np.concatenate([array.astype(dtype, copy=False) for array in arrays]) if arrays else np.empty(0, dtype=dtype)
    values_spec = type(spec)(
        name=spec.name,
        getter=spec.getter,
        axes=("value",),
        dynamic=spec.dynamic,
        optional=spec.optional,
        encoding="raggedValues",
    )
    offsets_spec = type(spec)(
        name=spec.name + "Offsets",
        getter=spec.getter,
        axes=("offset",),
        dynamic=spec.dynamic,
        optional=spec.optional,
        encoding="raggedOffsets",
    )
    return (
        _writeNumeric(iteration, io, path=path + "/values", typeName=typeName, spec=values_spec, value=values),
        _writeNumeric(iteration, io, path=path + "/offsets", typeName=typeName, spec=offsets_spec, value=offsets),
    )


def _selectedFields(graph: TransportGraph, *, dynamicOnly: bool):
    for node in graph.nodes:
        for name, (spec, value) in node.fields.items():
            if dynamicOnly and not spec.dynamic:
                continue
            if dynamicOnly:
                value = spec.value(node.owner)
                if value is None and not spec.optional:
                    raise ValueError(
                        f"required dynamic transport field {node.typeName}.{name} is None"
                    )
            yield node, name, spec, value


def writeGraph(iteration, graph: TransportGraph, io, *, dynamicOnly=False) -> None:
    """Write selected fields/references without inspecting concrete frontend types."""
    selectedFields = list(_selectedFields(graph, dynamicOnly=dynamicOnly))
    selectedNodes = (
        {node.path: node for node, _name, _spec, _value in selectedFields}
        if dynamicOnly
        else {node.path: node for node in graph.nodes}
    )
    iteration.set_attribute("haseTransportVersion", TRANSPORT_VERSION)
    iteration.set_attribute("haseUpdateMode", "dynamic" if dynamicOnly else "full")
    iteration.set_attribute("haseRoot", graph.root)
    iteration.set_attribute(
        "haseNodePaths",
        json.dumps(list(selectedNodes), separators=(",", ":")),
    )
    iteration.set_attribute(
        "haseNodeTypes",
        json.dumps(
            [node.typeName for node in selectedNodes.values()],
            separators=(",", ":"),
        ),
    )

    pending = []
    for node in selectedNodes.values():
        for name, paths in node.references.items():
            selectedPaths = [path for path in paths if path in selectedNodes]
            if dynamicOnly and not selectedPaths:
                continue
            iteration.set_attribute(
                referenceName(f"{node.path}/{name}"),
                json.dumps(selectedPaths, separators=(",", ":")),
            )
    for node, name, spec, value in selectedFields:
        if value is None:
            continue
        path = f"{node.path}/{name}"
        if spec.encoding == "ragged":
            pending.extend(
                _writeRagged(iteration, io, path=path, typeName=node.typeName, spec=spec, value=value)
            )
            continue
        if spec.encoding == "json":
            iteration.set_attribute(
                attributeName(path),
                json.dumps(value, separators=(",", ":"), sort_keys=True),
            )
            continue
        if isinstance(value, str) or (
            isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value)
        ):
            iteration.set_attribute(
                attributeName(path),
                value
                if isinstance(value, str)
                else json.dumps(list(value), separators=(",", ":")),
            )
            continue
        pending.append(
            _writeNumeric(iteration, io, path=path, typeName=node.typeName, spec=spec, value=value)
        )
    iteration.series_flush()
