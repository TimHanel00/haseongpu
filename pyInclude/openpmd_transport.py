from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .structures import ComputeParameters, ExperimentParameters, HostMesh, Result


DIMENSIONLESS = {}
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
    _prefer_matching_openpmd_api(find_calc_phi_ase())
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
    return {io.Unit_Dimension.L: 1.0}


def _as_array(values, dtype, shape=None, order="C"):
    arr = np.asarray(values, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape, order=order)
    return np.ascontiguousarray(arr)


def _reset_scalar_record(record, data, axis_labels):
    io = _io()
    record.set_attribute("geometry", "other")
    record.set_attribute("geometryParameters", "topology=unstructured_triangular_prism")
    record.set_attribute("dataOrder", "C")
    record.axis_labels = axis_labels
    record.grid_spacing = [1.0] * data.ndim
    record.grid_global_offset = [0.0] * data.ndim
    record.grid_unit_SI = 1.0
    record.unit_dimension = DIMENSIONLESS
    component = record[io.Mesh_Record_Component.SCALAR]
    component.unit_SI = 1.0
    component.position = [0.0] * data.ndim
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)


def _reset_component(record, component_name, data, axis_labels, unit_dimension):
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
    component.unit_SI = 1.0
    component.position = [0.0] * data.ndim
    component.reset_dataset(io.Dataset(data.dtype, data.shape))
    component.store_chunk(data)


def _load_scalar(series, iteration, name, dtype):
    io = _io()
    component = iteration.meshes[name][io.Mesh_Record_Component.SCALAR]
    chunk = component.load_chunk()
    series.flush()
    return np.array(chunk, dtype=dtype, copy=True).reshape(-1)


def find_calc_phi_ase():
    env = os.environ.get("HASE_CALCPHIASE")
    if env:
        path = Path(env)
        if path.is_file():
            return path

    root = Path(__file__).resolve().parents[1]
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


def write_input(path, experiment: ExperimentParameters, compute: ComputeParameters, mesh: HostMesh):
    path = Path(path)
    io = _io()
    series = io.Series(str(path), _access("create_linear"), _series_config(path))
    series.set_software("HASEonGPU-openPMD-python-frontend")

    iteration = series.snapshots()[0]
    iteration.time = 0.0
    iteration.dt = 1.0
    iteration.time_unit_SI = 1.0

    iteration.set_attribute("number_of_points", int(mesh.numberOfPoints))
    iteration.set_attribute("number_of_cells", int(mesh.numberOfTriangles))
    iteration.set_attribute("number_of_levels", int(mesh.numberOfLevels))
    iteration.set_attribute("thickness", float(mesh.thickness))
    iteration.set_attribute("n_tot", float(mesh.nTot))
    iteration.set_attribute("crystal_t_fluo", float(mesh.crystalTFluo))
    iteration.set_attribute("cladding_number", int(mesh.claddingNumber))
    iteration.set_attribute("cladding_absorption", float(mesh.claddingAbsorption))
    iteration.set_attribute("min_rays_per_sample", int(experiment.minRaysPerSample))
    iteration.set_attribute("max_rays_per_sample", int(experiment.maxRaysPerSample))
    iteration.set_attribute("mse_threshold", float(experiment.mseThreshold))
    iteration.set_attribute("repetitions", int(compute.maxRepetitions))
    iteration.set_attribute("adaptive_steps", int(compute.adaptiveSteps))
    iteration.set_attribute("use_reflections", bool(experiment.useReflections))
    iteration.set_attribute("spectral_resolution", int(experiment.spectral))
    iteration.set_attribute("monochromatic", bool(experiment.monochromatic))
    iteration.set_attribute("max_sigma_absorption", float(experiment.maxSigmaA))
    iteration.set_attribute("max_sigma_emission", float(experiment.maxSigmaE))
    iteration.set_attribute("backend", str(compute.backend))
    iteration.set_attribute("max_gpus", int(compute.maxGpus))
    iteration.set_attribute("parallel_mode", str(compute.parallelMode))
    iteration.set_attribute("min_sample_range", int(compute.minSampleRange))
    iteration.set_attribute("max_sample_range", int(compute.maxSampleRange))

    prefix = "core_"
    points = _as_array(mesh.points, np.float64, (2, int(mesh.numberOfPoints))).T
    _reset_component(iteration.meshes[prefix + "vertices"], "x", points[:, 0], ["point"], _length_dimension())
    _reset_component(iteration.meshes[prefix + "vertices"], "y", points[:, 1], ["point"], _length_dimension())
    _reset_scalar_record(
        iteration.meshes[prefix + "connectivity"],
        _as_array(mesh.trianglePointIndices, np.uint32, (3, int(mesh.numberOfTriangles))).T,
        ["cell", "local_vertex"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "neighbors"],
        _as_array(mesh.triangleNeighbors, np.int32, (3, int(mesh.numberOfTriangles))).T,
        ["cell", "local_side"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "forbidden_edges"],
        _as_array(mesh.forbiddenEdge, np.int32, (3, int(mesh.numberOfTriangles))).T,
        ["cell", "local_side"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "normal_points"],
        _as_array(mesh.triangleNormalPoint, np.uint32, (3, int(mesh.numberOfTriangles))).T,
        ["cell", "local_side"],
    )
    _reset_component(
        iteration.meshes[prefix + "cell_center"],
        "x",
        _as_array(mesh.triangleCenterX, np.float64),
        ["cell"],
        _length_dimension(),
    )
    _reset_component(
        iteration.meshes[prefix + "cell_center"],
        "y",
        _as_array(mesh.triangleCenterY, np.float64),
        ["cell"],
        _length_dimension(),
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "cell_normal_x"],
        _as_array(mesh.triangleNormalsX, np.float64, (3, int(mesh.numberOfTriangles))).T,
        ["cell", "local_side"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "cell_normal_y"],
        _as_array(mesh.triangleNormalsY, np.float64, (3, int(mesh.numberOfTriangles))).T,
        ["cell", "local_side"],
    )
    _reset_scalar_record(iteration.meshes[prefix + "surface"], _as_array(mesh.triangleSurfaces, np.float32), ["cell"])
    _reset_scalar_record(
        iteration.meshes[prefix + "beta_volume"],
        _as_array(mesh.betaVolume, np.float64, (int(mesh.numberOfLevels) - 1, int(mesh.numberOfTriangles))).T,
        ["cell", "layer"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "point_beta"],
        _as_array(mesh.betaCells, np.float64, (int(mesh.numberOfLevels), int(mesh.numberOfPoints))).T,
        ["point", "level"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "cladding_cell_type"],
        _as_array(mesh.claddingCellTypes, np.uint32),
        ["cell"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "refractive_index"],
        _as_array(mesh.refractiveIndices, np.float32),
        ["interface"],
    )
    _reset_scalar_record(
        iteration.meshes[prefix + "reflectivity"],
        _as_array(mesh.reflectivities, np.float32, (2, int(mesh.numberOfTriangles))).T,
        ["cell", "interface"],
    )
    _reset_scalar_record(iteration.meshes[prefix + "lambda_absorption"], _as_array(experiment.lambdaA, np.float64), ["wavelength"])
    _reset_scalar_record(iteration.meshes[prefix + "lambda_emission"], _as_array(experiment.lambdaE, np.float64), ["wavelength"])
    _reset_scalar_record(iteration.meshes[prefix + "sigma_absorption"], _as_array(experiment.sigmaA, np.float64), ["wavelength"])
    _reset_scalar_record(iteration.meshes[prefix + "sigma_emission"], _as_array(experiment.sigmaE, np.float64), ["wavelength"])

    iteration.close()
    series.close()


def read_result(path) -> Result:
    path = Path(path)
    series = _io().Series(str(path), _access("read_linear"), _series_config(path))
    for iteration in series.read_iterations():
        prefix = "core_result_"
        result = Result(
            phiAse=_load_scalar(series, iteration, prefix + "phi_ase", np.float32),
            mse=_load_scalar(series, iteration, prefix + "mse", np.float64),
            totalRays=_load_scalar(series, iteration, prefix + "total_rays", np.uint32),
            dndtAse=_load_scalar(series, iteration, prefix + "dndt_ase", np.float64),
        )
        iteration.close()
        series.close()
        return result
    raise RuntimeError(f"No result iteration was available in {path}")


def calcPhiASE(experiment: ExperimentParameters, compute: ComputeParameters, mesh: HostMesh, *, transport="sst"):
    suffix = ".sst" if transport == "sst" else ".bp"
    with tempfile.TemporaryDirectory(prefix="hase-openpmd-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / ("input" + suffix)
        output_path = tmp_path / ("output" + suffix)
        executable_path = find_calc_phi_ase()
        if transport == "sst":
            _prefer_matching_openpmd_api(executable_path)
        executable = str(executable_path)
        if transport == "sst":
            proc = subprocess.Popen([executable, f"--input-path={input_path}", f"--output-path={output_path}"])
            try:
                write_input(input_path, experiment, compute, mesh)
                result = read_result(output_path)
            finally:
                return_code = proc.wait()
        else:
            write_input(input_path, experiment, compute, mesh)
            completed = subprocess.run([executable, f"--input-path={input_path}", f"--output-path={output_path}"], check=False)
            return_code = completed.returncode
            if return_code != 0:
                raise RuntimeError(f"calcPhiASE failed with return code {return_code}")
            result = read_result(output_path)

        if return_code != 0:
            raise RuntimeError(f"calcPhiASE failed with return code {return_code}")
        return result
