from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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


SST_CONFIG = {
    "adios2": {
        "engine": {
            "parameters": {
                "DataTransport": "WAN",
                "OpenTimeoutSecs": "600",
                "QueueFullPolicy": "Discard",
            }
        }
    }
}


def _series_config(path: Path):
    return SST_CONFIG if path.suffix == ".sst" else {}


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
            "ADIOS2-SST requires the Python writer and C++ reader to use the same "
            "openPMD-api/ADIOS2 build. Restart Python with the CMake-built "
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


def findCalcPhiAse():
    env = os.environ.get("HASE_CALCPHIASE")
    if env:
        path = Path(env)
        if path.is_file():
            return path

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "calcPhiASE",
        root / "cmake-build-debug" / "calcPhiASE",
    ]
    for build_dir in root.glob("build/*"):
        candidates.append(build_dir / "calcPhiASE")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find calcPhiASE in the HASE build tree. Build the standard "
        "HASE target or set HASE_CALCPHIASE."
    )



def writeInput(path, phiAse, gainMedium, crossSections):
    path = Path(path)
    io = _io()
    series = io.Series(str(path), _access("create_linear"), _series_config(path))
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



def _runOpenPmdAndExecuteHaseBinary(phiAse, gainMedium, crossSections, *, transport="sst"):
    suffix = ".sst" if transport == "sst" else ".bp"
    with tempfile.TemporaryDirectory(prefix="hase-openpmd-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / ("input" + suffix)
        output_path = tmp_path / ("output" + suffix)
        executable_path = findCalcPhiAse()
        if transport == "sst":
            _prefer_matching_openpmd_api(executable_path)
        executable = str(executable_path)
        if transport == "sst":
            proc = subprocess.Popen([executable, f"--input-path={input_path}", f"--output-path={output_path}"], stderr=subprocess.PIPE, text=True)
            try:
                writeInput(input_path, phiAse, gainMedium, crossSections)
                result = read_result(output_path)
            finally:
                try:
                    return_code = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return_code = proc.wait()
            stderr = "" if proc.stderr is None else proc.stderr.read()
        else:
            writeInput(input_path, phiAse, gainMedium, crossSections)
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

        if return_code != 0:
            detail = f": {stderr.strip()}" if stderr and stderr.strip() else ""
            raise RuntimeError(f"calcPhiASE failed with return code {return_code}{detail}")
        return result


def runPhiASE(phiAse, gainMedium, crossSections, *, transport="sst"):
    return _runOpenPmdAndExecuteHaseBinary(phiAse, gainMedium, crossSections, transport=transport)
