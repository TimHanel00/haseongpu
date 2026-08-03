from __future__ import annotations

import contextlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..geometry import OpenPmdComponentField, OpenPmdScalarField
from .._runtime import runtime_config, runtime_executable_candidates, runtime_root
from .backends import _clean_backend_names, _load_backend_names
from . import (
    HASE_TRANSPORT_VERSION,
    FieldSpec,
    backendFlatArray,
    fieldSpec,
    flatEntityLabel,
    haseTransportAttributes,
    resultAttributeSpecs,
    resultFieldSpecs,
    simulationAttributeSpecs,
    spectralContext,
    unitDimension,
)
from ..structures import Result
from ..problem import AbsorbingSurface, ConstantReflectivitySurface
from hase_units import units


CANONICAL_POINTS_SPEC = FieldSpec(
    "canonicalPoints",
    "points",
    ("coordinate", "mesh_point"),
    np.float64,
    lambda context: (3, context.numberOfMeshPoints),
    unit="m",
    unitDimension=unitDimension.canonicalPoints,
    backendRequired=False,
)
CANONICAL_CONNECTIVITY_SPEC = FieldSpec(
    "canonicalConnectivity",
    "cells_connectivity",
    ("cell", "local_vertex"),
    np.uint32,
    lambda context: (context.numberOfCells, 4),
    unitDimension=unitDimension.canonicalConnectivity,
    backendRequired=False,
)
CANONICAL_OFFSETS_SPEC = FieldSpec(
    "canonicalOffsets",
    "cells_offsets",
    ("cell_offset",),
    np.uint32,
    lambda context: (context.numberOfCells + 1,),
    unitDimension=unitDimension.canonicalOffsets,
    backendRequired=False,
)
CANONICAL_CELL_TYPES_SPEC = FieldSpec(
    "canonicalCellTypes",
    "cells_types",
    ("cell",),
    np.uint32,
    lambda context: (context.numberOfCells,),
    unitDimension=unitDimension.canonicalCellTypes,
    backendRequired=False,
)
EXPLICIT_CELL_FACE_OFFSETS_SPEC = FieldSpec(
    "explicitCellFaceOffsets",
    "cell_face_offsets",
    ("cell_offset",),
    np.uint32,
    lambda context: (context.numberOfCells + 1,),
    backendRequired=False,
)
EXPLICIT_CELL_FACES_SPEC = FieldSpec(
    "explicitCellFaces",
    "cell_faces",
    ("cell", "local_face", "local_vertex"),
    np.int32,
    lambda context: (context.numberOfCells, context.numberOfFacesPerCell, 3),
    backendRequired=False,
)
EXPLICIT_CELL_NEIGHBORS_SPEC = FieldSpec(
    "explicitCellNeighbors",
    "cell_neighbor_cells",
    ("cell", "local_face"),
    np.int32,
    lambda context: (context.numberOfCells, context.numberOfFacesPerCell),
    backendRequired=False,
)
EXPLICIT_CELL_NEIGHBOR_FACES_SPEC = FieldSpec(
    "explicitCellNeighborFaces",
    "cell_neighbor_local_faces",
    ("cell", "local_face"),
    np.int32,
    lambda context: (context.numberOfCells, context.numberOfFacesPerCell),
    backendRequired=False,
)
EXPLICIT_FACE_BOUNDARIES_SPEC = FieldSpec(
    "explicitFaceBoundaries",
    "cell_face_boundaries",
    ("cell", "local_face"),
    np.int32,
    lambda context: (context.numberOfCells, context.numberOfFacesPerCell),
    backendRequired=False,
)
EXPLICIT_CELL_DOMAINS_SPEC = FieldSpec(
    "explicitCellDomains",
    "cell_domains",
    ("cell",),
    np.int32,
    lambda context: (context.numberOfCells,),
    backendRequired=False,
)
DYNAMIC_FIELD_NAMES = {"betaVolume"}
EXPLICIT_BETA_VOLUME_SPEC = FieldSpec(
    "betaVolume",
    "beta_volume",
    ("cell",),
    np.float64,
    lambda context: (context.numberOfCells,),
    dynamic=True,
)
def _env_flag(name):
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _runtime_config():
    return runtime_config()


def _forward_backend_logging_enabled():
    override = _env_flag("HASE_FORWARD_LOGGING")
    if override is not None:
        return override
    return bool(getattr(_runtime_config(), "HASE_FORWARD_LOGGING", False))


def _forward_backend_logging(stdout="", stderr=""):
    if not _forward_backend_logging_enabled():
        return
    if stdout:
        sys.stdout.write(stdout)
        sys.stdout.flush()
    if stderr:
        sys.stderr.write(stderr)
        sys.stderr.flush()


def _backend_failure_detail(stdout="", stderr=""):
    sections = []
    if stdout and stdout.strip():
        sections.append("calcPhiASE stdout:\n" + stdout.strip())
    if stderr and stderr.strip():
        sections.append("calcPhiASE stderr:\n" + stderr.strip())
    return ": " + "\n".join(sections) if sections else ""


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
            "type": "sst",
            "parameters": {
                "DataTransport": "WAN",
                "OpenTimeoutSecs": "600",
            }
        }
    },
}

_STREAMING_RESULT_EOF = object()
_STREAMING_RESULT_POLL_SECONDS = 0.1
_STREAMING_THREAD_JOIN_TIMEOUT_SECONDS = 10.0

OPENPMD_BACKENDS = {
    "adios": _BackendSpec("adios", ".bp", ADIOS2_CONFIG),
    "adios-sst": _BackendSpec("adios-sst", ".sst", SST_CONFIG, streaming=True),
    "hdf5": _BackendSpec("hdf5", ".h5", HDF5_CONFIG),
}
DEFAULT_OPENPMD_BACKEND = "auto"
OPENPMD_BACKEND_PRIORITY = ("adios", "adios-sst", "hdf5")
HASE_CONFIGURE_HINT = "Run `hase-configure` to generate a matching backend/openPMD setup."
_OPENPMD_BACKEND_PROBE_CACHE = {}


def _normalize_backend(backend=None):
    value = backend if backend is not None else DEFAULT_OPENPMD_BACKEND
    normalized = str(value).strip().lower()
    if normalized != "auto" and normalized not in OPENPMD_BACKENDS:
        allowed = ", ".join(("auto", *OPENPMD_BACKEND_PRIORITY))
        raise ValueError(f"unsupported openPMD backend '{value}'; expected one of: {allowed}. {HASE_CONFIGURE_HINT}")
    return normalized


def _backend_spec(backend=None):
    normalized = _normalize_backend(backend)
    if normalized == "auto":
        raise RuntimeError("openPMD backend 'auto' must be resolved against the installed runtime")
    return OPENPMD_BACKENDS[normalized]


def _probed_openpmd_backends(executable):
    executable = Path(executable).resolve()
    cache_key = str(executable)
    if cache_key in _OPENPMD_BACKEND_PROBE_CACHE:
        return _OPENPMD_BACKEND_PROBE_CACHE[cache_key]

    backends, probe_library = _load_backend_names(
        (
            executable.parent,
            executable.parent / "python" / "pyInclude" / "_runtime",
        )
    )
    backends = _clean_backend_names(backends)
    if not backends:
        raise RuntimeError(
            "openPMD backend-probe library did not report any supported backends. " + HASE_CONFIGURE_HINT
        )
    result = (backends, probe_library)
    _OPENPMD_BACKEND_PROBE_CACHE[cache_key] = result
    return result


def _ensure_compiled_backend_available(spec, executable):
    backends, probe_library = _probed_openpmd_backends(executable)
    if spec.name not in backends:
        available = ", ".join(backends)
        raise RuntimeError(
            f"openPMD backend '{spec.name}' is selected by MonteCarloASESolver.openpmd_backend "
            f"'openpmd_backend', but the runtime openPMD backend-probe library reports "
            f"available backends: {available}. Probe library: {probe_library}. Change "
            f"openpmd_backend in the YAML or fix the runtime openPMD provider environment. {HASE_CONFIGURE_HINT}"
        )


def _python_supported_backend_names(io):
    variants = getattr(io, "variants", {})
    extensions = set(getattr(io, "file_extensions", []))
    supported = []
    for name in OPENPMD_BACKEND_PRIORITY:
        spec = OPENPMD_BACKENDS[name]
        provider_enabled = variants.get("hdf5", False) if name == "hdf5" else variants.get("adios2", False)
        if provider_enabled and spec.suffix.lstrip(".") in extensions:
            supported.append(name)
    return tuple(supported)


def _resolve_backend(backend, executable):
    requested = _normalize_backend(backend)
    executable = Path(executable)
    io = _io(executable)
    compiled, probe_library = _probed_openpmd_backends(executable)
    python_backends = _python_supported_backend_names(io)
    supported = tuple(name for name in OPENPMD_BACKEND_PRIORITY if name in compiled and name in python_backends)
    if requested == "auto":
        if supported:
            return OPENPMD_BACKENDS[supported[0]]
        raise RuntimeError(
            "No compatible openPMD runtime backend was found. "
            f"Compiled provider supports: {', '.join(compiled)}; Python openPMD-api supports: "
            f"{', '.join(python_backends)}; probe library: {probe_library}. {HASE_CONFIGURE_HINT}"
        )
    spec = OPENPMD_BACKENDS[requested]
    if requested not in compiled:
        _ensure_compiled_backend_available(spec, executable)
    if requested not in python_backends:
        raise RuntimeError(
            f"openPMD backend '{requested}' is selected by MonteCarloASESolver.openpmd_backend "
            f"'openpmd_backend', but the active Python openPMD-api supports: "
            f"{', '.join(python_backends)}. {HASE_CONFIGURE_HINT}"
        )
    return spec


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _streaming_thread_join_timeout(value=None):
    if value is None:
        value = os.environ.get(
            "HASE_OPENPMD_THREAD_JOIN_TIMEOUT",
            str(_STREAMING_THREAD_JOIN_TIMEOUT_SECONDS),
        )
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("HASE_OPENPMD_THREAD_JOIN_TIMEOUT must be a positive number of seconds") from exc
    if seconds <= 0.0:
        raise ValueError("HASE_OPENPMD_THREAD_JOIN_TIMEOUT must be a positive number of seconds")
    return seconds


def _watchdog_interval(value=None):
    if value is None:
        value = os.environ.get("HASE_OPENPMD_WATCHDOG_INTERVAL", "30")
    text = str(value).strip().lower()
    if text in {"0", "none", "off", "false", "no"}:
        return None
    try:
        seconds = float(text)
    except ValueError as exc:
        raise ValueError("HASE_OPENPMD_WATCHDOG_INTERVAL must be a positive number of seconds, 0, or 'none'") from exc
    if seconds <= 0.0:
        raise ValueError("HASE_OPENPMD_WATCHDOG_INTERVAL must be a positive number of seconds, 0, or 'none'")
    return seconds


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


def _fields_from_domain(fields):
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


def _compiled_context(simulation):
    return _explicit_topology_context(simulation.mesh)


def _compiled_surface_optics(simulation):
    problem = simulation.resolvedProblem
    material = problem.materials[0]
    positive = np.asarray(simulation.mesh.surfaceDomainIds)
    positive = positive[positive > 0]
    size = int(np.max(positive, initial=-1)) + 1
    reflectivity = np.zeros(size, dtype=np.float32)
    inside = np.ones(size, dtype=np.float32)
    outside = np.ones(size, dtype=np.float32)
    for boundary_id, model in enumerate(problem.boundaries):
        domain_ids = np.unique(
            np.asarray(simulation.mesh.surfaceDomainIds)[problem.faceBoundaryId == boundary_id]
        )
        for identifier in domain_ids[domain_ids > 0]:
            identifier = int(identifier)
            if isinstance(model, AbsorbingSurface):
                reflectivity[identifier] = 0.0
                inside[identifier] = 1.0
                outside[identifier] = 1.0
            elif isinstance(model, ConstantReflectivitySurface):
                reflectivity[identifier] = np.float32(model.reflectivity)
                inside[identifier] = np.float32(material.refractiveIndex)
                outside[identifier] = np.float32(model.exterior_refractive_index)
            else:
                raise NotImplementedError(f"unsupported exterior boundary model {type(model).__name__}")
    return reflectivity, inside, outside


def _compiled_attribute_values(simulation):
    problem = simulation.resolvedProblem
    material = problem.materials[0]
    table = material.crossSections
    context = _compiled_context(simulation)
    sigma_a = np.asarray(table.absorption.toValue(units.cm**2), dtype=np.float64)
    sigma_e = np.asarray(table.emission.toValue(units.cm**2), dtype=np.float64)
    values = {
        "numberOfPoints": context.numberOfPoints,
        "numberOfTriangles": context.numberOfCells,
        "numberOfLevels": context.numberOfLevels,
        "thickness": context.thickness,
        "nTot": float(material.activeIonDensity.toValue(units.cm**-3)),
        "crystalTFluo": float(material.fluorescenceLifetime.toValue(units.s)),
        "claddingNumber": np.iinfo(np.uint32).max,
        "claddingAbsorption": 0.0,
        "maxSigmaAbsorption": float(np.max(sigma_a, initial=0.0)),
        "maxSigmaEmission": float(np.max(sigma_e, initial=0.0)),
    }
    values.update(simulation.aseSolver.transportAttributes(context.numberOfCells))
    return values


def _compiled_attribute_fields(simulation):
    values = _compiled_attribute_values(simulation)
    for spec in simulationAttributeSpecs:
        if spec.name not in values:
            if spec.name == "rngSeed":
                continue
            raise KeyError(spec.name)
        yield _AttributeField(spec.attribute, spec.cast(values[spec.name]))


def _compiled_array_fields(simulation):
    problem = simulation.resolvedProblem
    topology = simulation.mesh
    material = problem.materials[0]
    table = material.crossSections
    context = _compiled_context(simulation)
    excitation = np.asarray(problem.initialExcitationFraction, dtype=np.float64)
    if topology.numberOfSamplePoints == excitation.size:
        sampled = excitation
    elif np.all(excitation == excitation[0]):
        sampled = np.full(topology.numberOfSamplePoints, excitation[0], dtype=np.float64)
    else:
        raise NotImplementedError(
            "the compiled backend requires uniform initial excitation when sampling points differ from cells"
        )
    reflectivity, inside, outside = _compiled_surface_optics(simulation)
    fields = (
        OpenPmdScalarField(
            "betaCells",
            sampled,
            context,
            spec=FieldSpec(
                "betaCells",
                "point_beta",
                ("point", "level"),
                np.float64,
                lambda ctx: (ctx.numberOfPoints, ctx.numberOfLevels),
                dynamic=True,
            ),
        ),
        OpenPmdScalarField(
            "betaVolume",
            excitation,
            context,
            spec=FieldSpec(
                "betaVolume", "beta_volume", ("cell",), np.float64,
                lambda ctx: (ctx.numberOfCells,), dynamic=True,
            ),
        ),
        OpenPmdScalarField("claddingCellType", np.zeros(context.numberOfCells, dtype=np.uint32), context),
        OpenPmdScalarField(
            "refractiveIndex",
            np.asarray([material.refractiveIndex, 1.0, material.refractiveIndex, 1.0], dtype=np.float32),
            context,
        ),
        OpenPmdScalarField("reflectivity", np.zeros(context.numberOfCells * 2, dtype=np.float32), context),
        OpenPmdScalarField("surfaceReflectivity", reflectivity, context),
        OpenPmdScalarField("surfaceRefractiveIndexInside", inside, context),
        OpenPmdScalarField("surfaceRefractiveIndexOutside", outside, context),
        OpenPmdScalarField("lambdaAbsorption", table.wavelengths.toValue(units.nm), spectralContext(table.wavelengths.magnitude)),
        OpenPmdScalarField("lambdaEmission", table.wavelengths.toValue(units.nm), spectralContext(table.wavelengths.magnitude)),
        OpenPmdScalarField("sigmaAbsorption", table.absorption.toValue(units.cm**2), spectralContext(table.absorption.magnitude)),
        OpenPmdScalarField("sigmaEmission", table.emission.toValue(units.cm**2), spectralContext(table.emission.magnitude)),
    )
    yield from _fields_from_domain(fields)



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


def _io(executable=None):
    _prefer_matching_openpmd_api(find_calc_phi_ase() if executable is None else Path(executable))
    try:
        import openpmd_api as io
    except ImportError as exc:
        raise ImportError(
            "The openPMD transport requires an openpmd_api Python module matching "
            "the calcPhiASE/openPMD C++ stack. Install openpmd-api in this Python "
            "environment with the same backend/MPI options as the openPMD C++ "
            "package found by CMake, or build HASEonGPU with "
            "HASE_BUILD_OPENPMD_FROM_SOURCE=ON. " + HASE_CONFIGURE_HINT
        ) from exc
    return io


def _ensure_backend_available(backend, executable=None):
    executable = find_calc_phi_ase() if executable is None else Path(executable)
    return _resolve_backend(backend, executable)


def _openpmd_python_package_parent(path):
    path = Path(path)
    return path.parent if path.name == "openpmd_api" else path


def _env_openpmd_python_paths():
    for name in ("HASE_OPENPMD_PYTHONPATH", "HASE_OPENPMD_PYTHON_PACKAGE_DIR"):
        value = os.environ.get(name)
        if not value:
            continue
        for entry in value.split(os.pathsep):
            if entry:
                yield _openpmd_python_package_parent(Path(entry))


def _configured_openpmd_python_paths():
    configured = getattr(_runtime_config(), "HASE_OPENPMD_PYTHON_PACKAGE_DIR", "")
    if configured:
        yield _openpmd_python_package_parent(Path(configured))


def _using_external_openpmd():
    return bool(getattr(_runtime_config(), "HASE_USE_SYSTEM_OPENPMD", False))


def _candidate_python_paths(executable: Path):
    yield from _env_openpmd_python_paths()

    build_dir = executable.parent
    if (build_dir / "openpmd_api").is_dir():
        yield build_dir
    if (build_dir.parent / "openpmd_api").is_dir():
        yield build_dir.parent
    if (build_dir / "site-packages" / "openpmd_api").is_dir():
        yield build_dir / "site-packages"

    yield from _configured_openpmd_python_paths()

    yield from build_dir.glob("_deps/openpmd-build/lib/python*/site-packages")
    for parent in build_dir.parents:
        yield from parent.glob("_deps/openpmd-build/lib/python*/site-packages")


def _unique_existing_directories(paths):
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        yield resolved


def _prefer_matching_openpmd_api(executable: Path):
    candidates = list(_unique_existing_directories(_candidate_python_paths(executable)))
    if not candidates:
        if _using_external_openpmd():
            return
        raise RuntimeError(
            "The openPMD transport requires an openpmd_api Python module "
            "compatible with the openPMD C++ provider used by calcPhiASE. "
            "Install/load a matching provider, set "
            "-DHASE_OPENPMD_PYTHON_PACKAGE_DIR=<site-packages directory>, or set "
            "HASE_OPENPMD_PYTHONPATH at runtime. " + HASE_CONFIGURE_HINT
        )
    if "openpmd_api" in sys.modules:
        active = Path(getattr(sys.modules["openpmd_api"], "__file__", "")).resolve()
        for candidate in candidates:
            try:
                active.relative_to(candidate.resolve())
                return
            except ValueError:
                pass
        raise RuntimeError(
            "The openPMD transport requires the Python writer and C++ reader to use the same "
            "openPMD-api build/provider. Restart Python with the CMake-selected "
            "openpmd_api package first on PYTHONPATH, e.g. "
            f"PYTHONPATH={candidates[0]}:$PYTHONPATH. {HASE_CONFIGURE_HINT}"
        )

    for candidate in candidates:
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
    geometry_parameters="topology=explicit_tet4_volume",
):
    io = _io()
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", geometry_parameters)
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    # ADIOS2 SST may not preserve openPMD axisLabels on streamed mesh records.
    # Keep the canonical axisLabels property, plus a scalar fallback for readers.
    record.set_attribute("haseAxisLabelsString", ",".join(axis_labels))
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
    record.set_attribute("haseTransportVersion", HASE_TRANSPORT_VERSION)
    record.set_attribute("haseSchemaVersion", HASE_TRANSPORT_VERSION)
    record.set_attribute("haseEntity", spec.entity)
    record.set_attribute("haseAxes", list(spec.axes))
    # ADIOS2 SST has been observed to stream string-list attributes as an empty
    # scalar string. Keep haseAxes canonical, plus a scalar fallback.
    record.set_attribute("haseAxesString", ",".join(spec.axes))
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
    data = np.array(
        backendFlatArray(values, spec, context, layoutOrder="backendFlat"),
        copy=True,
        order="C",
    )
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
    record.set_attribute("geometryParameters", "topology=explicit_tet4_volume")
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    # ADIOS2 SST may not preserve openPMD axisLabels on streamed mesh records.
    # Keep the canonical axisLabels property, plus a scalar fallback for readers.
    record.set_attribute("haseAxisLabelsString", ",".join(axis_labels))
    record.grid_spacing = [1.0] * data.ndim
    record.grid_global_offset = [0.0] * data.ndim
    record.grid_unit_SI = 1.0
    record.unit_dimension = unit_dimension
    component = record[component_name]
    component.unit_SI = float(unit_si)
    component.position = [0.0] * data.ndim
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)


def _explicit_topology_context(topology):
    return SimpleNamespace(
        numberOfMeshPoints=topology.numberOfPoints,
        numberOfPoints=int(getattr(topology, "structuredNumberOfPoints", topology.numberOfSamplePoints)),
        numberOfCells=topology.numberOfCells,
        numberOfTriangles=topology.numberOfCells,
        numberOfLevels=int(getattr(topology, "structuredNumberOfLevels", 1)),
        thickness=float(getattr(topology, "structuredThickness", 0.0)),
        numberOfFacesPerCell=topology.numberOfFacesPerCell,
        numberOfCellVertices=4,
        numberOfSamplePoints=topology.numberOfSamplePoints,
        numberOfSurfaceDomains=int(np.max(topology.faceBoundaries[topology.faceBoundaries > 0]) + 1) if np.any(topology.faceBoundaries > 0) else 0,
    )


def _explicit_point_components(topology):
    points = np.asarray(topology.points, dtype=np.float64)
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
    }


def _write_explicit_static_topology(iteration, topology):
    context = _explicit_topology_context(topology)
    coordinate_unit = topology.coordinateUnit
    points_spec = replace(
        CANONICAL_POINTS_SPEC,
        unit=coordinate_unit.symbol,
        unitSI=coordinate_unit.unitSI,
        unitDimension=coordinate_unit.unitDimension,
    )
    record = iteration.meshes["core_" + points_spec.recordName]
    for component_name, values in _explicit_point_components(topology).items():
        _resetComponent(
            record,
            component_name,
            np.ascontiguousarray(values),
            ["mesh_point"],
            _unit_dimension(_io(), points_spec.unitDimension),
            points_spec.unitSI,
        )
    _record_metadata(record, points_spec)
    record.set_attribute("geometryParameters", "topology=explicit_tet4_volume")
    record.set_attribute("hasePrimitiveShape", list(points_spec.expectedShape(context)))

    def backend_flat(values):
        return np.asarray(values).reshape(-1)

    for spec, values in (
        (CANONICAL_CONNECTIVITY_SPEC, topology.cellsConnectivityFlat()),
        (CANONICAL_OFFSETS_SPEC, topology.cellsOffsets()),
        (CANONICAL_CELL_TYPES_SPEC, topology.cellTypes),
        (
            EXPLICIT_CELL_FACE_OFFSETS_SPEC,
            np.arange(context.numberOfCells + 1, dtype=np.uint32) * np.uint32(context.numberOfFacesPerCell),
        ),
        (EXPLICIT_CELL_FACES_SPEC, backend_flat(topology.facePointIndices)),
        (EXPLICIT_CELL_NEIGHBORS_SPEC, backend_flat(topology.neighborCells)),
        (EXPLICIT_CELL_NEIGHBOR_FACES_SPEC, backend_flat(topology.neighborLocalFaces)),
        (EXPLICIT_FACE_BOUNDARIES_SPEC, backend_flat(topology.faceBoundaries)),
        (EXPLICIT_CELL_DOMAINS_SPEC, topology.cellDomains),
    ):
        _resetFlatField(iteration.meshes["core_" + spec.recordName], spec, values, context)

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


def find_calc_phi_ase():
    env = os.environ.get("HASE_CALCPHIASE")
    if env:
        path = Path(env)
        if _is_openpmd_calc_phi_ase(path):
            return path
        raise RuntimeError(f"HASE_CALCPHIASE does not point to an openPMD calcPhiASE binary: {path}")

    root = runtime_root()
    for candidate in runtime_executable_candidates(("calcPhiASE",)):
        if _is_openpmd_calc_phi_ase(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find an openPMD calcPhiASE binary in the selected HASE runtime '{root}'. "
        "Build that runtime, set HASE_RUNTIME_DIR, or override the executable with "
        "HASE_CALCPHIASE. " + HASE_CONFIGURE_HINT
    )



def _open_input_series(path, *, backend=None):
    series = _io().Series(str(path), _access("create_linear"), _series_config(path, backend))
    series.set_software("HASEonGPU-openPMD-python-frontend")
    for name, value in haseTransportAttributes.items():
        series.set_attribute(name, value)
    series.set_attribute("haseTransportVersion", HASE_TRANSPORT_VERSION)
    series.set_attribute("haseSchemaVersion", HASE_TRANSPORT_VERSION)
    return series


def _write_compiled_input_iteration(series, iteration_index, simulation, *, run_control):
    iteration = series.snapshots()[int(iteration_index)]
    iteration.time = 0.0
    iteration.dt = float(simulation.timeStepSize.toValue(units.s))
    iteration.time_unit_SI = 1.0
    for field in _compiled_attribute_fields(simulation):
        iteration.set_attribute(field.name, field.value)
    for name, value in run_control.items():
        if value is not None:
            iteration.set_attribute(name, value)
    iteration.set_attribute("haseStaticUpdate", True)
    _write_explicit_static_topology(iteration, simulation.mesh)
    for field in _compiled_array_fields(simulation):
        _write_array_field(iteration, field)
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
            self.backend = _ensure_backend_available(self.backend).name
        self._series = _open_input_series(self.path, backend=self.backend)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def write_simulation(self, simulation, *, iteration_index=0, run_control):
        if self._series is None:
            raise RuntimeError("OpenPmdInputSeries must be used as a context manager before writing")
        index = int(iteration_index)
        _write_compiled_input_iteration(
            self._series,
            index,
            simulation,
            run_control=run_control,
        )
        self._series.flush()
        self._next_iteration = max(self._next_iteration, index + 1)
        return index

    def close(self):
        if self._series is not None:
            self._series.close()
            self._series = None




def _write_array_field(iteration, field):
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


def _read_optional_scalar(series, iteration, name, dtype, default_size=None):
    if name not in iteration.meshes:
        if default_size is None:
            return None
        return np.zeros(default_size, dtype=dtype)
    return _loadScalar(series, iteration, name, dtype)


def _has_attribute(obj, name):
    if hasattr(obj, "contains_attribute"):
        return obj.contains_attribute(name)
    try:
        obj.get_attribute(name)
        return True
    except Exception:
        return False


def _result_status_values(iteration):
    defaults = {
        "srmStatus": "disabled",
        "srmPasses": 0,
        "srmRemainingFraction": 0.0,
        "srmMaxIterations": 0,
        "srmDivergenceStreak": 0,
    }
    return {
        spec.name: spec.cast(iteration.get_attribute(spec.attribute)) if _has_attribute(iteration, spec.attribute) else defaults[spec.name]
        for spec in resultAttributeSpecs
    }


def read_simulation_output(path):
    """Read C++ time-stepped simulation snapshots from an openPMD output series."""
    path = Path(path)
    series = _io().Series(str(path), _access("read_linear"), _series_config(path))
    states = []
    for fallback_index, iteration in enumerate(series.read_iterations()):
        iteration_index = _iteration_index(iteration, fallback_index)
        number_of_points = int(iteration.get_attribute("number_of_points"))
        number_of_levels = int(iteration.get_attribute("number_of_levels"))
        number_of_cells = int(iteration.get_attribute("number_of_cells"))
        point_count = number_of_points * number_of_levels
        beta_cells = _loadScalar(series, iteration, "core_point_beta", np.float64)
        beta_volume = _loadScalar(series, iteration, "core_beta_volume", np.float64)
        phi_ase = _loadScalar(series, iteration, "core_result_phi_ase", np.float32)
        dndt_ase = _loadScalar(series, iteration, "core_result_dndt_ase", np.float64)
        volume_phi_ase = _read_optional_scalar(series, iteration, "core_result_volume_phi_ase", np.float32, number_of_cells)
        volume_standard_error = _read_optional_scalar(
            series, iteration, "core_result_volume_standard_error", np.float64, number_of_cells
        )
        volume_relative_standard_error = _read_optional_scalar(
            series, iteration, "core_result_volume_relative_standard_error", np.float64, number_of_cells
        )
        volume_total_rays = _read_optional_scalar(
            series, iteration, "core_result_volume_total_rays", np.uint32, number_of_cells
        )
        volume_dndt_ase = _read_optional_scalar(
            series, iteration, "core_result_volume_dndt_ase", np.float64, number_of_cells
        )
        beta_volume_shape = (number_of_cells,) if beta_volume.size == number_of_cells else (number_of_cells, number_of_levels - 1)
        result_shape = (number_of_points, number_of_levels)
        states.append(SimpleNamespace(
            iterationIndex=iteration_index,
            step=int(iteration.get_attribute("step_index")) if _has_attribute(iteration, "step_index") else iteration_index + 1,
            time=float(iteration.get_attribute("time")) if _has_attribute(iteration, "time") else float(iteration.time),
            betaCells=beta_cells.reshape(result_shape, order="F"),
            betaVolume=beta_volume.reshape(beta_volume_shape, order="F"),
            phiAse=phi_ase.reshape(result_shape, order="F"),
            volumePhiAse=volume_phi_ase.reshape((number_of_cells,), order="F"),
            volumeStandardError=volume_standard_error.reshape((number_of_cells,), order="F"),
            volumeRelativeStandardError=volume_relative_standard_error.reshape((number_of_cells,), order="F"),
            volumeTotalRays=volume_total_rays.reshape((number_of_cells,), order="F"),
            volumeDndtAse=volume_dndt_ase.reshape((number_of_cells,), order="F"),
            dndtAse=dndt_ase.reshape(result_shape, order="F"),
            dndtPump=_read_optional_scalar(
                series,
                iteration,
                "core_result_dndt_pump",
                np.float64,
                point_count,
            ).reshape((number_of_points, number_of_levels), order="F"),
            aseResult=Result(
                phiAse=_read_optional_scalar(series, iteration, "core_result_phi_ase", np.float32, point_count),
                standardError=_read_optional_scalar(
                    series, iteration, "core_result_standard_error", np.float64, point_count
                ),
                relativeStandardError=_read_optional_scalar(
                    series, iteration, "core_result_relative_standard_error", np.float64, point_count
                ),
                totalRays=_read_optional_scalar(series, iteration, "core_result_total_rays", np.uint32, point_count),
                dndtAse=_read_optional_scalar(series, iteration, "core_result_dndt_ase", np.float64, point_count),
                **_result_status_values(iteration),
            ),
            staticUpdate=bool(iteration.get_attribute("haseStaticUpdate")) if _has_attribute(iteration, "haseStaticUpdate") else iteration_index == 0,
        ))
        iteration.close()
    series.close()
    if not states:
        raise RuntimeError(f"No simulation iterations were available in {path}")
    return states


_TIME_INTEGRATORS = {
    "explicit-euler",
    "heun",
    "midpoint",
    "runge-kutta-4",
    "frozen-phi-ase-runge-kutta-4",
    "implicit-euler",
    "exponential-euler",
}


def _time_integrator_name(solver):
    if isinstance(solver, str):
        name = solver
    else:
        name = getattr(solver, "name", None)
    if name not in _TIME_INTEGRATORS:
        raise ValueError(
            "compiled Simulation supports time integrators: "
            + ", ".join(sorted(_TIME_INTEGRATORS))
        )
    return name


def _simulation_run_control(simulation, *, steps, pump_steps):
    if pump_steps is None:
        pump_steps = simulation.pumpSolver.maxSteps
    pump_steps_value = (2**32 - 1) if pump_steps is None else int(pump_steps)
    if pump_steps_value < 0:
        raise ValueError("pumpSteps must be non-negative")
    solver = simulation.timeIntegrator
    control = {
        "time_step": float(simulation.timeStepSize.toValue(units.s)),
        "number_of_steps": int(steps),
        "enable_ase": bool(simulation.enableAse),
        "pre_pump": bool(simulation.prePump),
        "pump_steps": pump_steps_value,
        "time_integrator": _time_integrator_name(solver),
        "pump_schema_version": 1,
        "pump_ray_count": int(simulation.pumpSolver.rayCount),
        "pump_rng_seed": int(simulation.pumpSolver.seed),
    }
    control.update(_general_pump_attributes(simulation))
    if hasattr(solver, "iterations"):
        control["implicit_iterations"] = int(solver.iterations)
    if hasattr(solver, "tolerance"):
        control["implicit_tolerance"] = float(solver.tolerance)
    return control


def _append_offset(values, offsets, additions):
    values.extend(additions)
    offsets.append(len(values))


def _general_pump_attributes(simulation):
    """Flatten the general pump graph into openPMD iteration attributes."""
    topology = simulation.mesh
    source_surfaces, source_surface_offsets = [], [0]
    spectrum_wavelengths, spectrum_weights, spectrum_sigma_a, spectrum_sigma_e = [], [], [], []
    spectrum_offsets = [0]
    angular_polar, angular_azimuthal, angular_weights, angular_offsets = [], [], [], [0]
    profile_kind, profile_radius_u, profile_radius_v, profile_exponent = [], [], [], []
    profile_center, profile_axis_u, profile_axis_v = [], [], []
    source_relay_offsets = [0]
    relay_exit_surfaces, relay_exit_offsets = [], [0]
    relay_entry_surfaces, relay_entry_offsets = [], [0]
    relay_flip_u, relay_flip_v, relay_rotation = [], [], []
    relay_offset, relay_tilt, relay_magnification, relay_transmission = [], [], [], []
    relay_count = 0

    for physical, injector, relays in simulation.pumpRegistrations:
        _append_offset(
            source_surfaces,
            source_surface_offsets,
            injector.surface.ids if hasattr(injector.surface, "ids") else np.unique(topology.surfaceDomainIds[injector.surface.mask()]),
        )
        wavelengths = np.asarray(physical.spectrum.wavelengths.toValue(units.m), dtype=np.float64)
        _append_offset(spectrum_wavelengths, spectrum_offsets, wavelengths.tolist())
        spectrum_weights.extend(np.asarray(physical.spectrum.weights, dtype=np.float64).tolist())
        table = simulation.resolvedProblem.materials[0].crossSections
        spectrum_sigma_a.extend(float(table.absorptionAt(value * units.m).toValue(units.cm**2)) for value in wavelengths)
        spectrum_sigma_e.extend(float(table.emissionAt(value * units.m).toValue(units.cm**2)) for value in wavelengths)
        _append_offset(
            angular_polar,
            angular_offsets,
            np.asarray(physical.angularDistribution.polarAngles, dtype=np.float64).tolist(),
        )
        angular_azimuthal.extend(np.asarray(physical.angularDistribution.azimuthalAngles, dtype=np.float64).tolist())
        angular_weights.extend(np.asarray(physical.angularDistribution.weights, dtype=np.float64).tolist())

        profile = physical.profile
        is_super_gaussian = getattr(profile, "kind", "uniform") == "super-gaussian"
        profile_kind.append(1 if is_super_gaussian else 0)
        profile_radius_u.append(float(profile.radiusU.toValue(topology.coordinateUnit)) if is_super_gaussian else 1.0)
        profile_radius_v.append(float(profile.radiusV.toValue(topology.coordinateUnit)) if is_super_gaussian else 1.0)
        profile_exponent.append(float(profile.exponent) if is_super_gaussian else 2.0)
        profile_center.extend(profile.center.toValue(topology.coordinateUnit) if is_super_gaussian else (0.0, 0.0, 0.0))
        profile_axis_u.extend(profile.axisU if is_super_gaussian else (1.0, 0.0, 0.0))
        profile_axis_v.extend(profile.axisV if is_super_gaussian else (0.0, 1.0, 0.0))

        for relay in relays:
            _append_offset(
                relay_exit_surfaces,
                relay_exit_offsets,
                np.unique(topology.surfaceDomainIds[relay.exitSurface.mask()]),
            )
            _append_offset(
                relay_entry_surfaces,
                relay_entry_offsets,
                np.unique(topology.surfaceDomainIds[relay.entrySurface.mask()]),
            )
            relay_flip_u.append(int(relay.flipU))
            relay_flip_v.append(int(relay.flipV))
            relay_rotation.append(float(relay.rotation))
            relay_offset.extend(float(value) for value in relay.offset)
            relay_tilt.extend(float(value) for value in relay.tilt)
            relay_magnification.append(float(relay.magnification))
            relay_transmission.append(float(relay.transmission))
            relay_count += 1
        source_relay_offsets.append(relay_count)

    attributes = {
        "pump_source_total_power": [float(physical.totalPower.toValue(units.W)) for physical, _injector, _relays in simulation.pumpRegistrations],
        "pump_source_surface_offsets": source_surface_offsets,
        "pump_source_surfaces": source_surfaces,
        "pump_spectrum_offsets": spectrum_offsets,
        "pump_spectrum_wavelengths": spectrum_wavelengths,
        "pump_spectrum_weights": spectrum_weights,
        "pump_spectrum_sigma_absorption": spectrum_sigma_a,
        "pump_spectrum_sigma_emission": spectrum_sigma_e,
        "pump_angular_offsets": angular_offsets,
        "pump_angular_polar": angular_polar,
        "pump_angular_azimuthal": angular_azimuthal,
        "pump_angular_weights": angular_weights,
        "pump_profile_kind": profile_kind,
        "pump_profile_radius_u": profile_radius_u,
        "pump_profile_radius_v": profile_radius_v,
        "pump_profile_exponent": profile_exponent,
        "pump_profile_center": profile_center,
        "pump_profile_axis_u": profile_axis_u,
        "pump_profile_axis_v": profile_axis_v,
        "pump_source_relay_offsets": source_relay_offsets,
        "pump_relay_exit_offsets": relay_exit_offsets,
        "pump_relay_exit_surfaces": relay_exit_surfaces,
        "pump_relay_entry_offsets": relay_entry_offsets,
        "pump_relay_entry_surfaces": relay_entry_surfaces,
        "pump_relay_flip_u": relay_flip_u,
        "pump_relay_flip_v": relay_flip_v,
        "pump_relay_rotation": relay_rotation,
        "pump_relay_offset": relay_offset,
        "pump_relay_tilt": relay_tilt,
        "pump_relay_magnification": relay_magnification,
        "pump_relay_transmission": relay_transmission,
    }
    unsigned_names = {
        "pump_source_surface_offsets", "pump_source_surfaces", "pump_spectrum_offsets",
        "pump_angular_offsets", "pump_profile_kind", "pump_source_relay_offsets",
        "pump_relay_exit_offsets", "pump_relay_exit_surfaces",
        "pump_relay_entry_offsets", "pump_relay_entry_surfaces",
        "pump_relay_flip_u", "pump_relay_flip_v",
    }
    for name in unsigned_names:
        attributes[name] = np.asarray(attributes[name], dtype=np.uint64)
    # openPMD backends do not portably represent zero-length attributes. Relay
    # offset arrays retain the zero-relay shape; empty relay payloads are omitted.
    for name in tuple(attributes):
        if name.startswith("pump_relay_") and np.asarray(attributes[name]).size == 0:
            del attributes[name]
    return attributes


def _write_simulation_input(input_path, spec, simulation, run_control, *, close_after=None):
    with OpenPmdInputSeries(input_path, backend=spec.name) as writer:
        writer.write_simulation(simulation, iteration_index=0, run_control=run_control)
        if close_after is not None:
            close_after.wait()


def _run_streaming_simulation(command, input_path, output_path, spec, simulation, run_control):
    """Run compiled simulation while Python threads exchange SST snapshots."""
    result_queue = queue.Queue(maxsize=1)
    input_queue = queue.Queue(maxsize=1)
    backend_finished = threading.Event()

    def read_output():
        try:
            result_queue.put((True, read_simulation_output(output_path)))
        except BaseException as exc:
            result_queue.put((False, exc))

    def write_input():
        try:
            _write_simulation_input(
                input_path,
                spec,
                simulation,
                run_control,
                close_after=backend_finished,
            )
            input_queue.put((True, None))
        except BaseException as exc:
            input_queue.put((False, exc))
            if proc.poll() is None:
                proc.kill()

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reader = threading.Thread(
        target=read_output,
        name="HASE compiled simulation snapshot receiver",
        daemon=True,
    )
    reader.start()
    writer = threading.Thread(
        target=write_input,
        name="HASE compiled simulation input sender",
        daemon=True,
    )
    writer.start()

    try:
        stdout, stderr = proc.communicate()
    finally:
        backend_finished.set()
    _forward_backend_logging(stdout=stdout, stderr=stderr)

    timeout = _streaming_thread_join_timeout()
    deadline = time.monotonic() + timeout
    writer.join(timeout=max(0.0, deadline - time.monotonic()))
    reader.join(timeout=max(0.0, deadline - time.monotonic()))

    input_result = None
    if not writer.is_alive():
        try:
            input_result = input_queue.get_nowait()
        except queue.Empty:
            if proc.returncode == 0:
                raise RuntimeError("openPMD simulation input sender stopped without reporting completion")
        if input_result is not None:
            input_ok, input_payload = input_result
            if not input_ok:
                raise RuntimeError("openPMD simulation input writer failed") from input_payload

    if proc.returncode != 0:
        detail = _backend_failure_detail(stdout=stdout, stderr=stderr)
        raise RuntimeError(f"calcPhiASE failed with return code {proc.returncode}{detail}")
    if writer.is_alive():
        raise RuntimeError(f"openPMD simulation input sender thread did not stop within {timeout:g} seconds")
    if reader.is_alive():
        raise RuntimeError(f"openPMD simulation snapshot receiver thread did not stop within {timeout:g} seconds")

    try:
        ok, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError("openPMD simulation snapshot receiver stopped without returning snapshots") from exc
    if not ok:
        raise payload
    return payload


def run_simulation(simulation, *, steps, pump_steps=None, transport=None, command_prefix=None, workspace_dir=None):
    """Run the full time-stepped Simulation in the compiled C++/alpaka backend."""
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")
    spec = _backend_spec(transport)
    _ensure_backend_available(spec.name)
    executable = find_calc_phi_ase()
    run_control = _simulation_run_control(simulation, steps=steps, pump_steps=pump_steps)

    artifact_root = _artifact_root()
    if artifact_root is None and workspace_dir is not None:
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    workspace_context = (
        tempfile.TemporaryDirectory(prefix="hase-openpmd-simulation-", dir=workspace_dir)
        if artifact_root is None
        else contextlib.nullcontext(artifact_root)
    )
    with workspace_context as tmp:
        tmp_path = Path(tmp)
        tmp_path.mkdir(parents=True, exist_ok=True)
        stem = _artifact_run_id() if artifact_root is not None else "simulation"
        input_path = tmp_path / f"{stem}-input{spec.suffix}"
        output_path = tmp_path / f"{stem}-output{spec.suffix}"

        command = [
            *(command_prefix or []),
            str(executable),
            f"--input-path={input_path}",
            f"--output-path={output_path}",
            "--cpp-control",
        ]

        if spec.streaming:
            return _run_streaming_simulation(command, input_path, output_path, spec, simulation, run_control)

        _write_simulation_input(input_path, spec, simulation, run_control)
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        _forward_backend_logging(stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode != 0:
            detail = _backend_failure_detail(stdout=completed.stdout, stderr=completed.stderr)
            raise RuntimeError(f"calcPhiASE failed with return code {completed.returncode}{detail}")
        return read_simulation_output(output_path)
