from __future__ import annotations

import contextlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..geometry import OpenPmdComponentField, OpenPmdScalarField
from . import HASE_SCHEMA_VERSION, FieldSpec, backendFlatArray, fieldSpec, flatEntityLabel, haseExtensionAttributes, resultFieldSpecs, simulationAttributeSpecs, spectralContext
from ..structures import Result


CANONICAL_POINTS_SPEC = FieldSpec(
    "canonicalPoints",
    "points",
    ("coordinate", "mesh_point"),
    np.float64,
    lambda context: (3, context.numberOfMeshPoints),
    unit="m",
    unitDimension=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    backendRequired=False,
)
CANONICAL_CONNECTIVITY_SPEC = FieldSpec(
    "canonicalConnectivity",
    "cells_connectivity",
    ("cell", "local_vertex"),
    np.uint32,
    lambda context: (context.numberOfPrisms, 6),
    backendRequired=False,
)
CANONICAL_OFFSETS_SPEC = FieldSpec(
    "canonicalOffsets",
    "cells_offsets",
    ("cell_offset",),
    np.uint32,
    lambda context: (context.numberOfPrisms + 1,),
    backendRequired=False,
)
CANONICAL_CELL_TYPES_SPEC = FieldSpec(
    "canonicalCellTypes",
    "cells_types",
    ("cell",),
    np.uint32,
    lambda context: (context.numberOfPrisms,),
    backendRequired=False,
)
DYNAMIC_FIELD_NAMES = {"betaVolume", "pointBeta"}


@dataclass(frozen=True)
class _AttributeField:
    name: str
    value: object


@dataclass(frozen=True)
class _ScalarArrayField:
    spec: FieldSpec
    values: object
    context: object
    prefix: str = "core_"


@dataclass(frozen=True)
class _ComponentArrayField:
    recordName: str
    spec: FieldSpec
    components: dict[str, object]
    axisLabels: list[str]
    context: object
    prefix: str = "core_"


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    suffix: str
    config: dict
    streaming: bool = False


ADIOS2_CONFIG = {"backend": "adios2"}
HDF5_CONFIG = {"backend": "hdf5"}
SST_CONFIG = {
    "backend": "adios2",
    "adios2": {
        "engine": {
            "parameters": {
                "DataTransport": "WAN",
                "OpenTimeoutSecs": "600",
                "QueueFullPolicy": "Discard",
            }
        }
    },
}

OPENPMD_BACKENDS = {
    "adios": _BackendSpec("adios", ".bp", ADIOS2_CONFIG),
    "adios-sst": _BackendSpec("adios-sst", ".sst", SST_CONFIG, streaming=True),
    "bp": _BackendSpec("bp", ".bp", {}),
    "hdf5": _BackendSpec("hdf5", ".h5", HDF5_CONFIG),
}


def _normalize_backend(backend=None):
    value = backend if backend is not None else os.environ.get("HASE_OPENPMD_BACKEND", "bp")
    normalized = str(value).strip().lower()
    if normalized in {"sst", "adiossst"}:
        normalized = "adios-sst"
    if normalized not in OPENPMD_BACKENDS:
        allowed = ", ".join(sorted(OPENPMD_BACKENDS))
        raise ValueError(f"unsupported openPMD backend '{value}'; expected one of: {allowed}")
    return normalized


def _backend_spec(backend=None):
    return OPENPMD_BACKENDS[_normalize_backend(backend)]


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _artifact_root():
    explicit = os.environ.get("HASE_OPENPMD_ARTIFACT_DIR")
    if explicit:
        return Path(explicit)
    if _truthy(os.environ.get("HASE_OPENPMD_KEEP_ARTIFACTS", "")):
        return Path.cwd() / "hase-openpmd-artifacts"
    return None


def _safe_artifact_name(value):
    allowed = []
    for char in str(value):
        allowed.append(char if char.isalnum() or char in {"-", "_", "."} else "-")
    return "".join(allowed).strip(".-_") or "transport"


def _artifact_run_id():
    explicit = os.environ.get("HASE_OPENPMD_ARTIFACT_RUN_ID")
    if explicit:
        return _safe_artifact_name(explicit)
    prefix = _safe_artifact_name(os.environ.get("HASE_OPENPMD_ARTIFACT_PREFIX", "transport"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}-{os.getpid()}"


def _write_openpmd_handle(handle_path: Path, series_path: Path):
    handle_path.write_text(series_path.name + "\n", encoding="utf-8")


def _write_artifact_manifest(path: Path, *, backend, input_path, output_path, input_handle, output_handle, status, return_code=None):
    lines = [
        f"backend={backend}",
        f"status={status}",
        f"input={input_path}",
        f"inputHandle={input_handle}",
        f"output={output_path}",
        f"outputHandle={output_handle}",
    ]
    if return_code is not None:
        lines.append(f"returnCode={return_code}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fieldContext(gainMedium):
    topology = gainMedium.topology
    topology._require_levels()
    if topology.thickness is None:
        raise ValueError("topology thickness is required before running a simulation")
    return SimpleNamespace(
        numberOfPoints=topology.numberOfPoints,
        numberOfTriangles=topology.numberOfTriangles,
        numberOfLevels=int(topology.levels),
    )


def _validatePhiAseTransportOptions(phiAse):
    if bool(phiAse.writeVtk):
        raise ValueError("PhiASE.writeVtk is not supported by the openPMD transport")
    if getattr(phiAse, "devices", None):
        raise ValueError("PhiASE.devices is not supported by the openPMD transport")


def _attributeFields(phiAse, gainMedium, crossSections):
    values = _attributeValues(phiAse, gainMedium, crossSections)
    for spec in simulationAttributeSpecs:
        if spec.name not in values:
            if spec.name == "rngSeed":
                continue
            raise KeyError(spec.name)
        yield _AttributeField(spec.attribute, spec.cast(values[spec.name]))


def _attributeValues(phiAse, gainMedium, crossSections):
    _validatePhiAseTransportOptions(phiAse)
    context = _fieldContext(gainMedium)
    number_of_samples = context.numberOfPoints * context.numberOfLevels
    values = {}
    values.update(gainMedium.topology.openPmdAttributes(context))
    values.update(gainMedium.openPmdAttributes(context))
    values.update(crossSections.openPmdAttributes())
    values.update(phiAse.openPmdAttributes(numberOfSamples=number_of_samples))
    return values

def _arrayFields(gainMedium, crossSections, *, include_static=True):
    context = _fieldContext(gainMedium)
    for field in _fieldsFromDomain(gainMedium.openPmdFields(context)):
        if include_static or field.spec.name in DYNAMIC_FIELD_NAMES:
            yield field
    if include_static:
        yield from _fieldsFromDomain(crossSections.openPmdFields(spectralContext))


def _fieldsFromDomain(fields):
    for field in fields:
        if isinstance(field, OpenPmdComponentField):
            spec = fieldSpec(field.name)
            yield _ComponentArrayField(
                recordName=field.recordName or spec.recordName,
                spec=spec,
                components=field.components,
                axisLabels=field.axisLabels,
                context=field.context,
                prefix=field.prefix,
            )
            continue
        if isinstance(field, OpenPmdScalarField):
            yield _ScalarArrayField(
                field.spec if field.spec is not None else fieldSpec(field.name),
                field.values,
                field.context,
                prefix=field.prefix,
            )
            continue
        name, values, context = field
        yield _ScalarArrayField(fieldSpec(name), values, context)



def _unit_dimension(io, exponents):
    labels = (
        io.Unit_Dimension.L,
        io.Unit_Dimension.M,
        io.Unit_Dimension.T,
        io.Unit_Dimension.I,
        io.Unit_Dimension.theta,
        io.Unit_Dimension.N,
        io.Unit_Dimension.J,
    )
    return {label: float(exponent) for label, exponent in zip(labels, exponents) if exponent != 0.0}


def _dimensionless_dimension():
    return {}


def _series_config(path: Path, backend=None):
    if backend is not None:
        return _backend_spec(backend).config
    if path.suffix == ".sst":
        return SST_CONFIG
    if path.suffix == ".h5":
        return HDF5_CONFIG
    return {}


def _io():
    _prefer_matching_openpmd_api(findCalcPhiAse())
    try:
        import openpmd_api as io
    except ImportError as exc:
        raise ImportError(
            "The openPMD transport requires an openpmd_api Python module matching "
            "the CMake-built calcPhiASE/openPMD stack."
        ) from exc
    return io


def _ensure_backend_available(backend):
    spec = _backend_spec(backend)
    io = _io()
    variants = getattr(io, "variants", {})
    extensions = set(getattr(io, "file_extensions", []))

    if spec.name == "hdf5":
        if not variants.get("hdf5", False):
            raise RuntimeError(
                "openPMD backend 'hdf5' requires openPMD-api built with HDF5 support"
            )
    else:
        if not variants.get("adios2", False):
            raise RuntimeError(
                f"openPMD backend '{spec.name}' requires openPMD-api built with ADIOS2 support"
            )

    extension = spec.suffix.lstrip(".")
    if extension not in extensions:
        raise RuntimeError(
            f"openPMD backend '{spec.name}' requires file extension '{extension}' "
            f"but this openPMD-api build reports: {sorted(extensions)}"
        )


def _candidate_python_paths(executable: Path):
    build_dir = executable.parent
    if (build_dir / "openpmd_api").is_dir():
        yield build_dir
    if (build_dir / "site-packages" / "openpmd_api").is_dir():
        yield build_dir / "site-packages"
    yield from build_dir.glob("_deps/openpmd-build/lib/python*/site-packages")
    for parent in build_dir.parents:
        yield from parent.glob("_deps/openpmd-build/lib/python*/site-packages")


def _prefer_matching_openpmd_api(executable: Path):
    if "openpmd_api" in sys.modules:
        active = Path(getattr(sys.modules["openpmd_api"], "__file__", "")).resolve()
        for candidate in _candidate_python_paths(executable):
            try:
                active.relative_to(candidate.resolve())
                return
            except ValueError:
                pass
        raise RuntimeError(
            "The openPMD transport requires the Python writer and C++ reader to use the same "
            "openPMD-api build. Restart Python with the CMake-built "
            "openPMD module first on PYTHONPATH, e.g. "
            f"PYTHONPATH={next(_candidate_python_paths(executable), '<openpmd-python-path>')}:$PYTHONPATH"
        )

    for candidate in _candidate_python_paths(executable):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            return


def _access(name):
    io = _io()
    if hasattr(io, "Access_Type"):
        return getattr(io.Access_Type, name)
    return getattr(io.Access, name)


def _length_dimension():
    io = _io()
    return _unit_dimension(io, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def _as_array(values, dtype, shape=None, order="C"):
    arr = np.asarray(values, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape, order=order)
    return np.ascontiguousarray(arr)


def _reset_scalar_record(
    record,
    data,
    axis_labels,
    unit_dimension=None,
    unit_si=1.0,
    grid_unit_si=1.0,
    grid_spacing=None,
    grid_global_offset=None,
    geometry_parameters="topology=unstructured_triangular_prism",
):
    io = _io()
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", geometry_parameters)
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    record.grid_spacing = [1.0] * data.ndim if grid_spacing is None else list(grid_spacing)
    record.grid_global_offset = [0.0] * data.ndim if grid_global_offset is None else list(grid_global_offset)
    record.grid_unit_SI = float(grid_unit_si)
    record.unit_dimension = _dimensionless_dimension() if unit_dimension is None else unit_dimension
    component = record[io.Mesh_Record_Component.SCALAR]
    component.unit_SI = float(unit_si)
    component.position = [0.0] * data.ndim
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)


def _record_metadata(record, spec: FieldSpec):
    record.set_attribute("haseSchemaVersion", HASE_SCHEMA_VERSION)
    record.set_attribute("haseEntity", spec.entity)
    record.set_attribute("haseAxes", list(spec.axes))
    record.set_attribute("haseLayoutOrder", "backendFlat")
    record.set_attribute("haseStatic", not spec.dynamic)
    record.set_attribute("haseDynamic", spec.dynamic)
    record.set_attribute("haseBackendRequired", spec.backendRequired)
    record.set_attribute("haseUnit", spec.unit)
    record.set_attribute("haseUserDefined", spec.userDefined)
    if spec.userDefined:
        record.set_attribute("haseUserFieldName", spec.name)


def _resetFlatField(record, spec: FieldSpec, values, context):
    io = _io()
    data = np.ascontiguousarray(backendFlatArray(values, spec, context, layoutOrder="backendFlat"))
    _reset_scalar_record(
        record,
        data,
        [flatEntityLabel(spec)],
        _unit_dimension(io, spec.unitDimension),
        spec.unitSI,
    )
    _record_metadata(record, spec)
    record.set_attribute("hasePrimitiveShape", list(spec.expectedShape(context)))


def _resetComponent(record, component_name, data, axis_labels, unit_dimension, unit_si=1.0):
    io = _io()
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", "topology=unstructured_triangular_prism")
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    record.grid_spacing = [1.0] * data.ndim
    record.grid_global_offset = [0.0] * data.ndim
    record.grid_unit_SI = 1.0
    record.unit_dimension = unit_dimension
    component = record[component_name]
    component.unit_SI = float(unit_si)
    component.position = [0.0] * data.ndim
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)


def _canonical_topology_context(topology):
    topology._require_levels()
    return SimpleNamespace(
        numberOfMeshPoints=topology.numberOfPoints * int(topology.levels),
        numberOfPrisms=topology.numberOfTriangles * (int(topology.levels) - 1),
    )


def _canonical_point_components(topology):
    points = np.asarray(topology.points, dtype=np.float64)
    z_values = topology.levelCoordinates()
    return {
        "x": np.tile(points[:, 0], z_values.size),
        "y": np.tile(points[:, 1], z_values.size),
        "z": np.repeat(z_values, topology.numberOfPoints),
    }


def _canonical_cell_connectivity(topology):
    triangles = np.asarray(topology.trianglePointIndices, dtype=np.uint32)
    rows = []
    for level in range(int(topology.levels) - 1):
        lower = level * topology.numberOfPoints
        upper = (level + 1) * topology.numberOfPoints
        for tri in triangles:
            ids = [int(vertex) for vertex in tri]
            rows.append([
                ids[0] + lower,
                ids[1] + lower,
                ids[2] + lower,
                ids[0] + upper,
                ids[1] + upper,
                ids[2] + upper,
            ])
    return np.asarray(rows, dtype=np.uint32).reshape(-1)


def _write_canonical_static_topology(iteration, gainMedium):
    topology = gainMedium.topology
    context = _canonical_topology_context(topology)
    record = iteration.meshes["core_" + CANONICAL_POINTS_SPEC.recordName]
    for component_name, values in _canonical_point_components(topology).items():
        _resetComponent(
            record,
            component_name,
            np.ascontiguousarray(values),
            ["mesh_point"],
            _unit_dimension(_io(), CANONICAL_POINTS_SPEC.unitDimension),
            CANONICAL_POINTS_SPEC.unitSI,
        )
    _record_metadata(record, CANONICAL_POINTS_SPEC)
    record.set_attribute("hasePrimitiveShape", list(CANONICAL_POINTS_SPEC.expectedShape(context)))

    connectivity = _canonical_cell_connectivity(topology)
    _resetFlatField(
        iteration.meshes["core_" + CANONICAL_CONNECTIVITY_SPEC.recordName],
        CANONICAL_CONNECTIVITY_SPEC,
        connectivity,
        context,
    )
    _resetFlatField(
        iteration.meshes["core_" + CANONICAL_OFFSETS_SPEC.recordName],
        CANONICAL_OFFSETS_SPEC,
        np.arange(context.numberOfPrisms + 1, dtype=np.uint32) * np.uint32(6),
        context,
    )
    _resetFlatField(
        iteration.meshes["core_" + CANONICAL_CELL_TYPES_SPEC.recordName],
        CANONICAL_CELL_TYPES_SPEC,
        np.full(context.numberOfPrisms, 13, dtype=np.uint32),
        context,
    )

def _loadScalar(series, iteration, name, dtype):
    io = _io()
    component = iteration.meshes[name][io.Mesh_Record_Component.SCALAR]
    chunk = component.load_chunk()
    series.flush()
    return np.array(chunk, dtype=dtype, copy=True).reshape(-1)


def _build_dir_for_executable(executable: Path):
    path = Path(executable).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "CMakeCache.txt").is_file():
            return parent
        if parent == Path.cwd().resolve():
            break
    return None


def _target_uses_openpmd_main(build_dir):
    if build_dir is None:
        return True
    manifests = [build_dir / name for name in ("build.ninja", "Makefile", "compile_commands.json")]
    existing = [path for path in manifests if path.is_file()]
    if not existing:
        return True
    return any("src/openpmd_main.cpp" in path.read_text(encoding="utf-8", errors="ignore") for path in existing)


def _is_openpmd_calc_phi_ase(executable: Path):
    return executable.is_file() and _target_uses_openpmd_main(_build_dir_for_executable(executable))


def findCalcPhiAse():
    env = os.environ.get("HASE_CALCPHIASE")
    if env:
        path = Path(env)
        if _is_openpmd_calc_phi_ase(path):
            return path
        raise RuntimeError(f"HASE_CALCPHIASE does not point to an openPMD calcPhiASE binary: {path}")

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "ci" / "calcPhiASE",
        root / "build" / "calcPhiASE",
        root / "cmake-build-debug" / "calcPhiASE",
    ]
    for build_dir in root.glob("build/*"):
        candidates.append(build_dir / "calcPhiASE")

    for candidate in candidates:
        if _is_openpmd_calc_phi_ase(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find an openPMD calcPhiASE binary in the HASE build tree. "
        "Build HASE_BUILD_PhiAse with src/openpmd_main.cpp or set HASE_CALCPHIASE."
    )



def _open_input_series(path, *, backend=None):
    series = _io().Series(str(path), _access("create_linear"), _series_config(path, backend))
    series.set_software("HASEonGPU-openPMD-python-frontend")
    for name, value in haseExtensionAttributes.items():
        series.set_attribute(name, value)
    series.set_attribute("haseSchemaVersion", HASE_SCHEMA_VERSION)
    return series


def _write_input_iteration(series, iteration_index, phiAse, gainMedium, crossSections, *, include_static=True):
    iteration = series.snapshots()[int(iteration_index)]
    iteration.time = 0.0
    iteration.dt = 1.0
    iteration.time_unit_SI = 1.0

    for field in _attributeFields(phiAse, gainMedium, crossSections):
        iteration.set_attribute(field.name, field.value)

    if include_static:
        iteration.set_attribute("haseStaticUpdate", True)
        _write_canonical_static_topology(iteration, gainMedium)
    else:
        iteration.set_attribute("haseStaticUpdate", False)

    for field in _arrayFields(gainMedium, crossSections, include_static=include_static):
        _writeArrayField(iteration, field)

    iteration.close()


class OpenPmdInputSeries:
    """Context manager for writing HASE input iterations to one openPMD series."""

    def __init__(self, path, *, backend=None):
        self.path = Path(path)
        self.backend = backend
        self._series = None
        self._next_iteration = 0

    def __enter__(self):
        if self.backend is not None:
            _ensure_backend_available(self.backend)
        self._series = _open_input_series(self.path, backend=self.backend)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def write(self, phiAse, gainMedium, crossSections, *, iteration_index=None, include_static=None):
        if self._series is None:
            raise RuntimeError("OpenPmdInputSeries must be used as a context manager before writing")
        index = self._next_iteration if iteration_index is None else int(iteration_index)
        write_static = (index == 0) if include_static is None else bool(include_static)
        _write_input_iteration(self._series, index, phiAse, gainMedium, crossSections, include_static=write_static)
        self._series.flush()
        self._next_iteration = max(self._next_iteration, index + 1)
        return index

    def close(self):
        if self._series is not None:
            self._series.close()
            self._series = None




def _writeArrayField(iteration, field):
    if isinstance(field, _ComponentArrayField):
        record = iteration.meshes[field.prefix + field.recordName]
        for component_name, values in field.components.items():
            data = np.ascontiguousarray(values)
            _resetComponent(
                record,
                component_name,
                data,
                field.axisLabels,
                _unit_dimension(_io(), field.spec.unitDimension),
                field.spec.unitSI,
            )
        _record_metadata(record, field.spec)
        record.set_attribute("hasePrimitiveShape", list(field.spec.expectedShape(field.context)))
        return

    _resetFlatField(
        iteration.meshes[field.prefix + field.spec.recordName],
        field.spec,
        field.values,
        field.context,
    )


def _iteration_index(iteration, fallback=None):
    for name in ("iteration_index", "iterationIndex"):
        if hasattr(iteration, name):
            return int(getattr(iteration, name))
    return fallback


def _read_result_iteration(series, iteration, *, fallback_index=None) -> tuple[int | None, Result]:
    iteration_index = _iteration_index(iteration, fallback_index)
    prefix = "core_result_"
    values = {
        spec.name: _loadScalar(series, iteration, prefix + spec.recordName, spec.dtypeObject)
        for spec in resultFieldSpecs()
    }
    iteration.close()
    return iteration_index, Result(**values)


def read_result(path, *, expected_iteration_index=0) -> Result:
    path = Path(path)
    series = _io().Series(str(path), _access("read_linear"), _series_config(path))
    for fallback_index, iteration in enumerate(series.read_iterations()):
        iteration_index, result = _read_result_iteration(series, iteration, fallback_index=fallback_index)
        series.close()
        if iteration_index is not None and iteration_index != expected_iteration_index:
            raise RuntimeError(
                f"Expected result iteration {expected_iteration_index} in {path}, got {iteration_index}"
            )
        return result
    series.close()
    raise RuntimeError(f"No result iteration was available in {path}")


class OpenPmdPhiAseSession:
    """Run PhiASE requests through openPMD and wait for matching result iterations."""

    def __init__(self, *, transport=None, timeout=30, command_prefix=None, workspace_dir=None):
        self.spec = _backend_spec(transport)
        self.timeout = timeout
        self.command_prefix = [] if command_prefix is None else list(command_prefix)
        self.workspace_dir = None if workspace_dir is None else Path(workspace_dir)
        self._workspace = None
        self._tmp_path = None
        self._manifest_path = None
        self._input_handle = None
        self._output_handle = None
        self._input_path = None
        self._output_path = None
        self._executable = None
        self._proc = None
        self._input_series = None
        self._result_queue = None
        self._reader = None
        self._pending_results = {}
        self._next_iteration = 0
        self._entered = False

    def __enter__(self):
        artifact_root = _artifact_root()
        if artifact_root is None and self.workspace_dir is not None:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._workspace = (
            tempfile.TemporaryDirectory(prefix="hase-openpmd-", dir=self.workspace_dir)
            if artifact_root is None
            else contextlib.nullcontext(artifact_root)
        )
        tmp = self._workspace.__enter__()
        self._tmp_path = Path(tmp)
        self._tmp_path.mkdir(parents=True, exist_ok=True)
        self._executable = findCalcPhiAse()
        _ensure_backend_available(self.spec.name)

        artifact_id = _artifact_run_id() if artifact_root is not None else None
        if self.spec.streaming:
            stem = f"{artifact_id}-" if artifact_id else ""
            self._input_path = self._tmp_path / f"{stem}input{self.spec.suffix}"
            self._output_path = self._tmp_path / f"{stem}output{self.spec.suffix}"
            self._manifest_path = None if artifact_root is None else self._tmp_path / f"{artifact_id}-manifest.txt"
            self._input_handle = None if artifact_root is None else self._tmp_path / f"{artifact_id}-input.pmd"
            self._output_handle = None if artifact_root is None else self._tmp_path / f"{artifact_id}-output.pmd"
            self._write_handles_and_manifest(status="created")
            self._start_streaming_backend()

        self._entered = True
        return self

    def _calc_phi_ase_command(self, input_path, output_path):
        return [
            *self.command_prefix,
            str(self._executable),
            f"--input-path={input_path}",
            f"--output-path={output_path}",
        ]

    def __exit__(self, exc_type, exc, traceback):
        close_error = None
        try:
            self.close()
        except BaseException as error:
            close_error = error
        finally:
            self._entered = False
            if self._workspace is not None:
                self._workspace.__exit__(exc_type, exc, traceback)
                self._workspace = None
        if exc_type is None and close_error is not None:
            raise close_error
        return False

    def run(self, phiAse, gainMedium, crossSections):
        if not self._entered:
            raise RuntimeError("OpenPmdPhiAseSession must be used as a context manager before running")
        iteration_index = self._next_iteration
        if self.spec.streaming:
            result = self._run_streaming_iteration(iteration_index, phiAse, gainMedium, crossSections)
        else:
            result = self._run_file_iteration(iteration_index, phiAse, gainMedium, crossSections)
        self._next_iteration += 1
        return result

    def close(self):
        if self._input_series is not None:
            self._input_series.close()
            self._input_series = None
        if self._proc is None:
            return

        try:
            return_code = self._proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            return_code = self._proc.wait()
        stderr = "" if self._proc.stderr is None else self._proc.stderr.read()
        self._proc = None
        self._write_handles_and_manifest(status="completed" if return_code == 0 else "failed", return_code=return_code)
        if return_code != 0:
            detail = f": {stderr.strip()}" if stderr and stderr.strip() else ""
            raise RuntimeError(f"calcPhiASE failed with return code {return_code}{detail}")

    def _paths_for_file_iteration(self, iteration_index):
        artifact_root = _artifact_root()
        artifact_id = _artifact_run_id() if artifact_root is not None else None
        if artifact_id is None:
            if iteration_index == 0:
                return self._tmp_path / ("input" + self.spec.suffix), self._tmp_path / ("output" + self.spec.suffix), None
            return (
                self._tmp_path / f"input-{iteration_index}{self.spec.suffix}",
                self._tmp_path / f"output-{iteration_index}{self.spec.suffix}",
                None,
            )
        stem = f"{artifact_id}-{iteration_index}"
        return (
            self._tmp_path / f"{stem}-input{self.spec.suffix}",
            self._tmp_path / f"{stem}-output{self.spec.suffix}",
            self._tmp_path / f"{stem}-manifest.txt",
        )

    def _run_file_iteration(self, iteration_index, phiAse, gainMedium, crossSections):
        input_path, output_path, manifest_path = self._paths_for_file_iteration(iteration_index)
        input_handle = None
        output_handle = None
        if manifest_path is not None:
            input_handle = manifest_path.with_name(manifest_path.stem + "-input.pmd")
            output_handle = manifest_path.with_name(manifest_path.stem + "-output.pmd")
            _write_openpmd_handle(input_handle, input_path)
            _write_openpmd_handle(output_handle, output_path)
            _write_artifact_manifest(
                manifest_path,
                backend=self.spec.name,
                input_path=input_path,
                output_path=output_path,
                input_handle=input_handle,
                output_handle=output_handle,
                status="created",
            )

        with OpenPmdInputSeries(input_path, backend=self.spec.name) as writer:
            writer.write(phiAse, gainMedium, crossSections, iteration_index=iteration_index, include_static=True)
        completed = subprocess.run(
            self._calc_phi_ase_command(input_path, output_path),
            check=False,
            text=True,
            capture_output=True,
        )
        if manifest_path is not None:
            _write_artifact_manifest(
                manifest_path,
                backend=self.spec.name,
                input_path=input_path,
                output_path=output_path,
                input_handle=input_handle,
                output_handle=output_handle,
                status="completed" if completed.returncode == 0 else "failed",
                return_code=completed.returncode,
            )
        if completed.returncode != 0:
            detail = f": {completed.stderr.strip()}" if completed.stderr and completed.stderr.strip() else ""
            raise RuntimeError(f"calcPhiASE failed with return code {completed.returncode}{detail}")
        return read_result(output_path, expected_iteration_index=iteration_index)

    def _write_handles_and_manifest(self, *, status, return_code=None):
        if self._manifest_path is None:
            return
        _write_openpmd_handle(self._input_handle, self._input_path)
        _write_openpmd_handle(self._output_handle, self._output_path)
        _write_artifact_manifest(
            self._manifest_path,
            backend=self.spec.name,
            input_path=self._input_path,
            output_path=self._output_path,
            input_handle=self._input_handle,
            output_handle=self._output_handle,
            status=status,
            return_code=return_code,
        )

    def _start_streaming_backend(self):
        self._proc = subprocess.Popen(
            self._calc_phi_ase_command(self._input_path, self._output_path),
            stderr=subprocess.PIPE,
            text=True,
        )
        self._result_queue = queue.Queue()
        self._reader = threading.Thread(target=self._read_streaming_results, daemon=True)
        self._reader.start()
        self._input_series = _open_input_series(self._input_path, backend=self.spec.name)

    def _read_streaming_results(self):
        try:
            series = _io().Series(str(self._output_path), _access("read_linear"), _series_config(self._output_path))
            for fallback_index, iteration in enumerate(series.read_iterations()):
                self._result_queue.put((True, _read_result_iteration(series, iteration, fallback_index=fallback_index)))
            series.close()
        except BaseException as exc:
            self._result_queue.put((False, exc))

    def _run_streaming_iteration(self, iteration_index, phiAse, gainMedium, crossSections):
        _write_input_iteration(self._input_series, iteration_index, phiAse, gainMedium, crossSections, include_static=(iteration_index == 0))
        self._input_series.flush()
        return self._wait_for_result(iteration_index)

    def _wait_for_result(self, expected_iteration_index):
        if expected_iteration_index in self._pending_results:
            return self._pending_results.pop(expected_iteration_index)
        while True:
            if self._proc is not None and self._proc.poll() not in (None, 0):
                stderr = "" if self._proc.stderr is None else self._proc.stderr.read()
                detail = f": {stderr.strip()}" if stderr and stderr.strip() else ""
                raise RuntimeError(f"calcPhiASE failed with return code {self._proc.returncode}{detail}")
            try:
                ok, payload = self._result_queue.get(timeout=self.timeout)
            except queue.Empty as exc:
                if self._proc is not None:
                    self._proc.kill()
                raise TimeoutError(
                    f"Timed out waiting for openPMD backend '{self.spec.name}' result iteration "
                    f"{expected_iteration_index}"
                ) from exc
            if not ok:
                raise payload
            iteration_index, result = payload
            if iteration_index is None or iteration_index == expected_iteration_index:
                return result
            self._pending_results[iteration_index] = result


def _runOpenPmdAndExecuteHaseBinary(
    phiAse,
    gainMedium,
    crossSections,
    *,
    transport=None,
    command_prefix=None,
    workspace_dir=None,
):
    kwargs = {"transport": transport}
    if command_prefix is not None:
        kwargs["command_prefix"] = command_prefix
    if workspace_dir is not None:
        kwargs["workspace_dir"] = workspace_dir
    with OpenPmdPhiAseSession(**kwargs) as session:
        return session.run(phiAse, gainMedium, crossSections)

def runPhiASE(phiAse, gainMedium, crossSections, *, transport=None, command_prefix=None, workspace_dir=None):
    return _runOpenPmdAndExecuteHaseBinary(
        phiAse,
        gainMedium,
        crossSections,
        transport=transport,
        command_prefix=command_prefix,
        workspace_dir=workspace_dir,
    )
