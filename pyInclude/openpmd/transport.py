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
    if normalized == "sst":
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

def _arrayFields(gainMedium, crossSections):
    context = _fieldContext(gainMedium)
    yield from _fieldsFromDomain(gainMedium.topology.openPmdFields(context))
    yield from _fieldsFromDomain(gainMedium.openPmdFields(context))
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


def _reset_scalar_record(record, data, axis_labels, unit_dimension=None, unit_si=1.0, grid_unit_si=1.0):
    io = _io()
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", "topology=unstructured_triangular_prism")
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    record.grid_spacing = [1.0] * data.ndim
    record.grid_global_offset = [0.0] * data.ndim
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



def writeInput(path, phiAse, gainMedium, crossSections, *, backend=None):
    path = Path(path)
    if backend is not None:
        _ensure_backend_available(backend)
    io = _io()
    series = io.Series(str(path), _access("create_linear"), _series_config(path, backend))
    series.set_software("HASEonGPU-openPMD-python-frontend")
    for name, value in haseExtensionAttributes.items():
        series.set_attribute(name, value)
    series.set_attribute("haseSchemaVersion", HASE_SCHEMA_VERSION)

    iteration = series.snapshots()[0]
    iteration.time = 0.0
    iteration.dt = 1.0
    iteration.time_unit_SI = 1.0

    for field in _attributeFields(phiAse, gainMedium, crossSections):
        iteration.set_attribute(field.name, field.value)

    for field in _arrayFields(gainMedium, crossSections):
        _writeArrayField(iteration, field)

    iteration.close()
    series.close()




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


def read_result(path) -> Result:
    path = Path(path)
    series = _io().Series(str(path), _access("read_linear"), _series_config(path))
    for iteration in series.read_iterations():
        prefix = "core_result_"
        values = {
            spec.name: _loadScalar(series, iteration, prefix + spec.recordName, spec.dtypeObject)
            for spec in resultFieldSpecs()
        }
        iteration.close()
        series.close()
        return Result(**values)
    raise RuntimeError(f"No result iteration was available in {path}")



def _runOpenPmdAndExecuteHaseBinary(phiAse, gainMedium, crossSections, *, transport=None):
    spec = _backend_spec(transport)
    artifact_root = _artifact_root()
    workspace = tempfile.TemporaryDirectory(prefix="hase-openpmd-") if artifact_root is None else contextlib.nullcontext(artifact_root)
    with workspace as tmp:
        tmp_path = Path(tmp)
        if artifact_root is not None:
            tmp_path.mkdir(parents=True, exist_ok=True)
            artifact_id = _artifact_run_id()
            input_path = tmp_path / f"{artifact_id}-input{spec.suffix}"
            output_path = tmp_path / f"{artifact_id}-output{spec.suffix}"
            input_handle = tmp_path / f"{artifact_id}-input.pmd"
            output_handle = tmp_path / f"{artifact_id}-output.pmd"
            manifest_path = tmp_path / f"{artifact_id}-manifest.txt"
            _write_openpmd_handle(input_handle, input_path)
            _write_openpmd_handle(output_handle, output_path)
            _write_artifact_manifest(
                manifest_path,
                backend=spec.name,
                input_path=input_path,
                output_path=output_path,
                input_handle=input_handle,
                output_handle=output_handle,
                status="created",
            )
        else:
            input_path = tmp_path / ("input" + spec.suffix)
            output_path = tmp_path / ("output" + spec.suffix)
            manifest_path = None
            input_handle = None
            output_handle = None
        executable_path = findCalcPhiAse()
        _ensure_backend_available(spec.name)
        executable = str(executable_path)
        if spec.streaming:
            proc = subprocess.Popen(
                [executable, f"--input-path={input_path}", f"--output-path={output_path}"],
                stderr=subprocess.PIPE,
                text=True,
            )
            result_queue = queue.Queue(maxsize=1)

            def read_streaming_result():
                try:
                    result_queue.put((True, read_result(output_path)))
                except BaseException as exc:
                    result_queue.put((False, exc))

            reader = threading.Thread(target=read_streaming_result, daemon=True)
            reader.start()
            try:
                writeInput(input_path, phiAse, gainMedium, crossSections, backend=spec.name)
                try:
                    ok, payload = result_queue.get(timeout=30)
                except queue.Empty as exc:
                    proc.kill()
                    raise TimeoutError(f"Timed out waiting for openPMD backend '{spec.name}' result stream") from exc
                if ok:
                    result = payload
                else:
                    raise payload
            finally:
                try:
                    return_code = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return_code = proc.wait()
            stderr = "" if proc.stderr is None else proc.stderr.read()
        else:
            writeInput(input_path, phiAse, gainMedium, crossSections, backend=spec.name)
            completed = subprocess.run(
                [executable, f"--input-path={input_path}", f"--output-path={output_path}"],
                check=False,
                text=True,
                capture_output=True,
            )
            return_code = completed.returncode
            stderr = completed.stderr
            if return_code == 0:
                result = read_result(output_path)

        if manifest_path is not None:
            _write_artifact_manifest(
                manifest_path,
                backend=spec.name,
                input_path=input_path,
                output_path=output_path,
                input_handle=input_handle,
                output_handle=output_handle,
                status="completed" if return_code == 0 else "failed",
                return_code=return_code,
            )
        if return_code != 0:
            detail = f": {stderr.strip()}" if stderr and stderr.strip() else ""
            raise RuntimeError(f"calcPhiASE failed with return code {return_code}{detail}")
        return result


def runPhiASE(phiAse, gainMedium, crossSections, *, transport=None):
    return _runOpenPmdAndExecuteHaseBinary(phiAse, gainMedium, crossSections, transport=transport)
