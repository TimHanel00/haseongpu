import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import pyInclude.openpmd.transport as transport
from pyInclude.geometry import GainMedium, MeshTopology
from pyInclude.laser import CrossSectionData
from pyInclude.openpmd import HASE_SCHEMA_VERSION, PrimitiveFieldSpec, PrismSchema, backendFlat, backendFlatArray, fieldSpec, haseExtensionAttributes, primitiveView, spectralContext
from pyInclude.simulation import PhiASE


pytest.importorskip("pyInclude.openpmd.transport")


MESH_FIELD_VALUES = {
    "points": np.array([0.0, 1.5, 0.25, 2.25, -0.75, 0.0, -0.5, 1.25, 2.5, 3.75], dtype=np.float64),
    "connectivity": np.array([0, 2, 0, 1, 3, 2, 2, 4, 4], dtype=np.uint32),
    "neighbors": np.array([-1, 0, 1, 1, -1, 0, 2, 2, -1], dtype=np.int32),
    "forbiddenEdges": np.array([-1, -1, 2, 0, 2, -1, 1, -1, 0], dtype=np.int32),
    "normalPoints": np.array([4, 2, 0, 3, 1, 2, 0, 4, 3], dtype=np.uint32),
    "cellCenterX": np.array([0.5, 1.25, -0.25], dtype=np.float64),
    "cellCenterY": np.array([0.75, 1.5, 2.25], dtype=np.float64),
    "cellNormalX": np.array([0.10, 0.40, 0.70, 0.20, 0.50, 0.80, 0.30, 0.60, 0.90], dtype=np.float64),
    "cellNormalY": np.array([-0.10, -0.40, -0.70, -0.20, -0.50, -0.80, -0.30, -0.60, -0.90], dtype=np.float64),
    "surface": np.array([1.25, 2.50, 3.75], dtype=np.float32),
    "betaVolume": np.array([0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33, 0.14, 0.24, 0.34, 0.15, 0.25, 0.35], dtype=np.float64),
    "pointBeta": np.array([100.0 + 10.0 * point + level for level in range(6) for point in range(5)], dtype=np.float64),
    "claddingCellType": np.array([0, 2, 1], dtype=np.uint32),
    "refractiveIndex": np.array([1.80, 1.20, 1.65, 1.05], dtype=np.float32),
    "reflectivity": np.array([0.01, 0.03, 0.05, 0.02, 0.04, 0.06], dtype=np.float32),
}

SPECTRAL_FIELD_VALUES = {
    "lambdaAbsorption": np.array([900e-9, 910e-9, 930e-9], dtype=np.float64),
    "lambdaEmission": np.array([1000e-9, 1015e-9, 1040e-9], dtype=np.float64),
    "sigmaAbsorption": np.array([0.010, 0.025, 0.040], dtype=np.float64),
    "sigmaEmission": np.array([0.050, 0.035, 0.020], dtype=np.float64),
}

SCALAR_RECORD_SPECS = {
    "connectivity": "connectivity",
    "neighbors": "neighbors",
    "forbidden_edges": "forbiddenEdges",
    "normal_points": "normalPoints",
    "cell_normal_x": "cellNormalX",
    "cell_normal_y": "cellNormalY",
    "surface": "surface",
    "beta_volume": "betaVolume",
    "point_beta": "pointBeta",
    "cladding_cell_type": "claddingCellType",
    "refractive_index": "refractiveIndex",
    "reflectivity": "reflectivity",
    "lambda_absorption": "lambdaAbsorption",
    "lambda_emission": "lambdaEmission",
    "sigma_absorption": "sigmaAbsorption",
    "sigma_emission": "sigmaEmission",
}


def asymmetric_topology():
    points = np.column_stack((MESH_FIELD_VALUES["points"][:5], MESH_FIELD_VALUES["points"][5:]))
    triangles = MESH_FIELD_VALUES["connectivity"].reshape((3, 3), order="F")
    return MeshTopology(points, triangles, levels=6, thickness=0.375)


def asymmetric_medium():
    return GainMedium(asymmetric_topology()).withPhysicalProperties(
        betaVolume=backendFlat(MESH_FIELD_VALUES["betaVolume"]),
        betaCells=backendFlat(MESH_FIELD_VALUES["pointBeta"]),
        claddingCellTypes=MESH_FIELD_VALUES["claddingCellType"],
        refractiveIndices=MESH_FIELD_VALUES["refractiveIndex"],
        reflectivities=backendFlat(MESH_FIELD_VALUES["reflectivity"]),
        nTot=7.5,
        crystalTFluo=1.75,
        claddingNumber=3,
        claddingAbsorption=0.075,
    )


def asymmetric_cross_sections():
    return CrossSectionData(
        wavelengthsAbsorption=SPECTRAL_FIELD_VALUES["lambdaAbsorption"],
        crossSectionAbsorption=SPECTRAL_FIELD_VALUES["sigmaAbsorption"],
        wavelengthsEmission=SPECTRAL_FIELD_VALUES["lambdaEmission"],
        crossSectionEmission=SPECTRAL_FIELD_VALUES["sigmaEmission"],
        resolution=3,
    )


def asymmetric_phi_ase():
    return PhiASE(
        crossSections=asymmetric_cross_sections(),
        minRaysPerSample=1,
        maxRaysPerSample=1,
        mseThreshold=0.25,
        repetitions=1,
        adaptiveSteps=1,
        useReflections=True,
        backend="Host_Cpu_CpuSerial",
        parallelMode="single",
        numDevices=1,
        minSampleRange=0,
        maxSampleRange=0,
        rngSeed=1234,
    )


def asymmetric_mesh():
    medium = asymmetric_medium()
    topology = medium.topology
    derived = topology._topology()
    return SimpleNamespace(
        numberOfPoints=topology.numberOfPoints,
        numberOfTriangles=topology.numberOfTriangles,
        numberOfLevels=int(topology.levels),
        points=np.asarray(topology.points).reshape(-1, order="F"),
        trianglePointIndices=np.asarray(topology.trianglePointIndices).reshape(-1, order="F"),
        triangleNeighbors=derived["triangleNeighbors"],
        forbiddenEdge=derived["forbiddenEdge"],
        triangleNormalPoint=derived["triangleNormalPoint"],
        triangleCenterX=derived["triangleCenterX"],
        triangleCenterY=derived["triangleCenterY"],
        triangleNormalsX=derived["triangleNormalsX"],
        triangleNormalsY=derived["triangleNormalsY"],
        triangleSurfaces=derived["triangleSurfaces"],
        betaVolume=medium.get("betaVolume").value,
        betaCells=medium.get("betaCells").value,
        claddingCellTypes=medium.get("claddingCellTypes").value,
        refractiveIndices=medium.get("refractiveIndices").value,
        reflectivities=medium.get("reflectivities").value,
    )


def _mesh_field_values(mesh):
    return {
        "points": np.asarray(mesh.points),
        "connectivity": np.asarray(mesh.trianglePointIndices),
        "neighbors": np.asarray(mesh.triangleNeighbors),
        "forbiddenEdges": np.asarray(mesh.forbiddenEdge),
        "normalPoints": np.asarray(mesh.triangleNormalPoint),
        "cellCenterX": np.asarray(mesh.triangleCenterX),
        "cellCenterY": np.asarray(mesh.triangleCenterY),
        "cellNormalX": np.asarray(mesh.triangleNormalsX),
        "cellNormalY": np.asarray(mesh.triangleNormalsY),
        "surface": np.asarray(mesh.triangleSurfaces),
        "betaVolume": np.asarray(mesh.betaVolume),
        "pointBeta": np.asarray(mesh.betaCells),
        "claddingCellType": np.asarray(mesh.claddingCellTypes),
        "refractiveIndex": np.asarray(mesh.refractiveIndices),
        "reflectivity": np.asarray(mesh.reflectivities),
    }


def _scalar_record_values(mesh):
    values = _mesh_field_values(mesh)
    return {
        "connectivity": values["connectivity"],
        "neighbors": values["neighbors"],
        "forbidden_edges": values["forbiddenEdges"],
        "normal_points": values["normalPoints"],
        "cell_normal_x": values["cellNormalX"],
        "cell_normal_y": values["cellNormalY"],
        "surface": values["surface"],
        "beta_volume": values["betaVolume"],
        "point_beta": values["pointBeta"],
        "cladding_cell_type": values["claddingCellType"],
        "refractive_index": values["refractiveIndex"],
        "reflectivity": values["reflectivity"],
        "lambda_absorption": SPECTRAL_FIELD_VALUES["lambdaAbsorption"],
        "lambda_emission": SPECTRAL_FIELD_VALUES["lambdaEmission"],
        "sigma_absorption": SPECTRAL_FIELD_VALUES["sigmaAbsorption"],
        "sigma_emission": SPECTRAL_FIELD_VALUES["sigmaEmission"],
    }


@pytest.fixture
def contract_input(tmp_path):
    output = tmp_path / "contract.bp"
    transport.writeInput(output, asymmetric_phi_ase(), asymmetric_medium(), asymmetric_cross_sections())
    return output


def _io():
    return transport._io()


def _read_scalar(series, iteration, name):
    io = _io()
    component = iteration.meshes[name][io.Mesh_Record_Component.SCALAR]
    chunk = component.load_chunk()
    series.flush()
    return np.array(chunk, copy=True)


def _read_component(series, component):
    chunk = component.load_chunk()
    series.flush()
    return np.array(chunk, copy=True)


def _attribute_list(value):
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _record_metadata(record, spec):
    return {
        "schema": record.get_attribute("haseSchemaVersion"),
        "entity": record.get_attribute("haseEntity"),
        "axes": _attribute_list(record.get_attribute("haseAxes")),
        "layout": record.get_attribute("haseLayoutOrder"),
        "primitive_shape": _attribute_list(record.get_attribute("hasePrimitiveShape")),
        "static": record.get_attribute("haseStatic"),
        "dynamic": record.get_attribute("haseDynamic"),
        "backend_required": record.get_attribute("haseBackendRequired"),
        "unit": record.get_attribute("haseUnit"),
        "axis_labels": list(record.axis_labels),
        "unit_si": record[_io().Mesh_Record_Component.SCALAR].unit_SI,
    }


def _unit_dimension_values(record, io):
    value = record.unit_dimension
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    labels = (
        io.Unit_Dimension.L,
        io.Unit_Dimension.M,
        io.Unit_Dimension.T,
        io.Unit_Dimension.I,
        io.Unit_Dimension.theta,
        io.Unit_Dimension.N,
        io.Unit_Dimension.J,
    )
    return tuple(float(value.get(label, 0.0)) for label in labels)


def _assert_hase_metadata(record, spec, context):
    assert _record_metadata(record, spec) == {
        "schema": HASE_SCHEMA_VERSION,
        "entity": spec.entity,
        "axes": list(spec.axes),
        "layout": "backendFlat",
        "primitive_shape": list(spec.expectedShape(context)),
        "static": not spec.dynamic,
        "dynamic": spec.dynamic,
        "backend_required": spec.backendRequired,
        "unit": spec.unit,
        "axis_labels": ["flatIndex"],
        "unit_si": spec.unitSI,
    }
    assert _unit_dimension_values(record, _io()) == spec.unitDimension


def _context_for_spec(spec_name):
    if spec_name in SPECTRAL_FIELD_VALUES:
        return spectralContext(SPECTRAL_FIELD_VALUES[spec_name])
    return asymmetric_mesh()


def test_layout_helpers_define_exact_backend_flat_contract():
    mesh = asymmetric_mesh()
    for name, values in _mesh_field_values(mesh).items():
        spec = fieldSpec(name)
        flat = backendFlatArray(values, spec, mesh, layoutOrder="backendFlat")
        np.testing.assert_array_equal(flat, values.astype(spec.dtypeObject, copy=False))
        view = primitiveView(backendFlat(values), spec, mesh)
        assert view.shape == spec.expectedShape(mesh)
        assert view.dtype == spec.dtypeObject
        np.testing.assert_array_equal(view.reshape(-1, order="F"), flat)

    for name, values in SPECTRAL_FIELD_VALUES.items():
        spec = fieldSpec(name)
        context = spectralContext(values)
        flat = backendFlatArray(values, spec, context, layoutOrder="backendFlat")
        np.testing.assert_array_equal(flat, values.astype(spec.dtypeObject, copy=False))
        assert primitiveView(backendFlat(values), spec, context).shape == spec.expectedShape(context)


def test_layout_helpers_reject_accidental_transpose_views():
    mesh = asymmetric_mesh()
    spec = fieldSpec("betaVolume")
    primitive = primitiveView(backendFlat(MESH_FIELD_VALUES["betaVolume"]), spec, mesh)
    transposed_same_size = np.asfortranarray(primitive.T)
    with pytest.raises(ValueError, match="expects primitive shape"):
        backendFlatArray(transposed_same_size, spec, mesh)


def test_writeInput_openpmd_contains_exact_values_order_shape_dtype_units_and_metadata(contract_input):
    io = _io()
    series = io.Series(str(contract_input), io.Access.read_only)
    for name, value in haseExtensionAttributes.items():
        assert series.get_attribute(name) == value
    assert series.get_attribute("haseSchemaVersion") == HASE_SCHEMA_VERSION
    iteration = series.iterations[0]
    mesh = asymmetric_mesh()

    assert iteration.get_attribute("number_of_points") == 5
    assert iteration.get_attribute("number_of_cells") == 3
    assert iteration.get_attribute("number_of_levels") == 6
    assert iteration.get_attribute("thickness") == pytest.approx(0.375)
    assert iteration.get_attribute("n_tot") == pytest.approx(7.5)
    assert iteration.get_attribute("crystal_t_fluo") == pytest.approx(1.75)
    assert iteration.get_attribute("spectral_resolution") == 3
    assert iteration.get_attribute("rng_seed") == 1234

    vertices = iteration.meshes["core_vertices"]
    assert vertices.get_attribute("haseSchemaVersion") == HASE_SCHEMA_VERSION
    assert vertices.get_attribute("haseEntity") == "coordinate_point"
    assert _attribute_list(vertices.get_attribute("haseAxes")) == ["coordinate", "point"]
    assert _attribute_list(vertices.get_attribute("hasePrimitiveShape")) == [2, 5]
    assert vertices.get_attribute("haseUnit") == "m"
    assert vertices.unit_dimension[io.Unit_Dimension.L] == 1.0
    np.testing.assert_array_equal(_read_component(series, vertices["x"]), MESH_FIELD_VALUES["points"][:5])
    np.testing.assert_array_equal(_read_component(series, vertices["y"]), MESH_FIELD_VALUES["points"][5:])
    series.flush()

    cell_center = iteration.meshes["core_cell_center"]
    np.testing.assert_array_equal(_read_component(series, cell_center["x"]), np.asarray(mesh.triangleCenterX))
    np.testing.assert_array_equal(_read_component(series, cell_center["y"]), np.asarray(mesh.triangleCenterY))
    assert cell_center.get_attribute("haseSchemaVersion") == HASE_SCHEMA_VERSION
    assert cell_center.get_attribute("haseEntity") == "cell"
    assert _attribute_list(cell_center.get_attribute("haseAxes")) == ["cell"]
    assert cell_center.get_attribute("haseLayoutOrder") == "backendFlat"
    assert _attribute_list(cell_center.get_attribute("hasePrimitiveShape")) == [3]
    assert cell_center.get_attribute("haseStatic") is True
    assert cell_center.get_attribute("haseDynamic") is False
    assert cell_center.get_attribute("haseBackendRequired") is True
    assert cell_center.get_attribute("haseUnit") == "m"
    assert list(cell_center.axis_labels) == ["cell"]
    assert cell_center.unit_dimension[io.Unit_Dimension.L] == 1.0
    series.flush()

    for record_name, spec_name in SCALAR_RECORD_SPECS.items():
        spec = fieldSpec(spec_name)
        context = _context_for_spec(spec_name)
        record = iteration.meshes["core_" + record_name]
        values = _read_scalar(series, iteration, "core_" + record_name)
        expected = _scalar_record_values(mesh)[record_name].astype(spec.dtypeObject, copy=False)
        assert values.shape == (expected.size,)
        assert values.dtype == spec.dtypeObject
        np.testing.assert_array_equal(values, expected)
        _assert_hase_metadata(record, spec, context)

    series.close()



def test_python_writer_openpmd_cpp_parser_result_round_trip(contract_input, tmp_path):
    output = tmp_path / "round_trip_result.bp"
    env = os.environ.copy()
    env["HASE_OPENPMD_PYTHON_CONTRACT_INPUT"] = str(contract_input)
    env["HASE_OPENPMD_PYTHON_CONTRACT_OUTPUT"] = str(output)
    helper = "cmake-build-debug/tests/tests_openpmdParserValidation"
    if not os.path.exists(helper):
        helper = "build/tests/tests_openpmdParserValidation"
    completed = subprocess.run(
        [helper, "openPMD parser round-trips a Python writer contract input"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    result = transport.read_result(output)
    expected_phi = np.array([0.5 + i for i in range(30)], dtype=np.float32)
    expected_mse = np.array([1000.0 + i for i in range(30)], dtype=np.float64)
    expected_total_rays = np.array([200 + i for i in range(30)], dtype=np.uint32)
    expected_dndt_ase = np.array([-10.0 - i for i in range(30)], dtype=np.float64)
    np.testing.assert_array_equal(result.phiAse, expected_phi)
    np.testing.assert_array_equal(result.mse, expected_mse)
    np.testing.assert_array_equal(result.totalRays, expected_total_rays)
    np.testing.assert_array_equal(result.dndtAse, expected_dndt_ase)

    io = _io()
    series = io.Series(str(output), io.Access.read_only)
    iteration = series.iterations[0]
    expected_units = {
        "phi_ase": "cm^-2 s^-1",
        "mse": "1",
        "total_rays": "count",
        "dndt_ase": "s^-1",
    }
    for name in ["phi_ase", "mse", "total_rays", "dndt_ase"]:
        record = iteration.meshes["core_result_" + name]
        assert record.get_attribute("haseSchemaVersion") == HASE_SCHEMA_VERSION
        assert record.get_attribute("haseEntity") == "point_level"
        assert _attribute_list(record.get_attribute("haseAxes")) == ["point", "level"]
        assert record.get_attribute("haseLayoutOrder") == "recordC"
        assert _attribute_list(record.get_attribute("hasePrimitiveShape")) == [5, 6]
        assert record.get_attribute("haseUnit") == expected_units[name]
        assert list(record.axis_labels) == ["point", "level"]
    series.close()



def _integration_enabled():
    return os.environ.get("HASE_RUN_OPENPMD_INTEGRATION", "").lower() in {"1", "true", "yes", "on"}


def _cache_value(cache_path, name):
    if cache_path is None or not cache_path.is_file():
        return None
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(name + ":"):
            return line.split("=", 1)[1].strip()
    return None


def _build_dir_for_executable(executable):
    path = Path(executable).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "CMakeCache.txt").is_file():
            return parent
        if parent == Path.cwd().resolve():
            break
    return None


def _target_uses_openpmd_main(build_dir):
    if build_dir is None:
        return False
    for manifest in ("build.ninja", "Makefile", "compile_commands.json"):
        path = build_dir / manifest
        if path.is_file() and "src/openpmd_main.cpp" in path.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def _calc_phi_ase_candidates():
    env = os.environ.get("HASE_OPENPMD_CALCPHIASE") or os.environ.get("HASE_CALCPHIASE")
    if env:
        yield Path(env)
    root = Path(__file__).resolve().parents[3]
    yield root / "cmake-build-debug" / "calcPhiASE"
    yield root / "build" / "calcPhiASE"
    yield root / "build" / "cp312-cp312-linux_x86_64" / "calcPhiASE"


def _openpmd_calc_phi_ase_or_skip():
    if not _integration_enabled():
        pytest.skip("set HASE_RUN_OPENPMD_INTEGRATION=1 to run calcPhiASE round-trip integration tests")
    for executable in _calc_phi_ase_candidates():
        if executable.is_file() and os.access(executable, os.X_OK):
            build_dir = _build_dir_for_executable(executable)
            if _target_uses_openpmd_main(build_dir):
                return executable.resolve(), build_dir
    pytest.skip("no openPMD calcPhiASE binary found; build HASE_BUILD_PhiAse with src/openpmd_main.cpp")


def _mpi_enabled_or_skip(build_dir):
    cache = None if build_dir is None else build_dir / "CMakeCache.txt"
    disable_mpi = _cache_value(cache, "DISABLE_MPI")
    if disable_mpi == "ON":
        pytest.skip("calcPhiASE was built with DISABLE_MPI=ON")
    compile_commands = None if build_dir is None else build_dir / "compile_commands.json"
    if compile_commands is not None and compile_commands.is_file():
        commands = compile_commands.read_text(encoding="utf-8", errors="ignore")
        if "src/openpmd_main.cpp" in commands and "-DDISABLE_MPI" in commands:
            pytest.skip("calcPhiASE compile command defines DISABLE_MPI")
    mpi_found = _cache_value(cache, "MPI_CXX_FOUND") or _cache_value(cache, "MPI_FOUND")
    if mpi_found is not None and mpi_found.upper() not in {"TRUE", "ON", "1", "YES"}:
        pytest.skip("calcPhiASE build did not find MPI")


def _round_trip_calc_phi_ase(tmp_path, parallel_mode):
    executable, build_dir = _openpmd_calc_phi_ase_or_skip()
    if parallel_mode == "mpi":
        _mpi_enabled_or_skip(build_dir)

    phi_ase = asymmetric_phi_ase()
    phi_ase.parallelMode = parallel_mode
    input_path = tmp_path / f"{parallel_mode}_input.bp"
    output_path = tmp_path / f"{parallel_mode}_output.bp"

    transport.writeInput(input_path, phi_ase, asymmetric_medium(), asymmetric_cross_sections())
    completed = subprocess.run(
        [str(executable), f"--input-path={input_path}", f"--output-path={output_path}"],
        check=False,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    result = transport.read_result(output_path)
    expected_size = asymmetric_topology().numberOfPoints * asymmetric_topology().levels
    assert result.phiAse.shape == (expected_size,)
    assert result.mse.shape == (expected_size,)
    assert result.totalRays.shape == (expected_size,)
    assert result.dndtAse.shape == (expected_size,)
    assert np.all(np.isfinite(result.phiAse))
    assert np.all(np.isfinite(result.mse))
    assert np.all(result.totalRays >= 0)

    io = _io()
    series = io.Series(str(input_path), io.Access.read_only)
    try:
        assert series.iterations[0].get_attribute("parallel_mode") == parallel_mode
    finally:
        series.close()


@pytest.mark.integration
def test_calc_phi_ase_single_openpmd_round_trip(tmp_path):
    _round_trip_calc_phi_ase(tmp_path, "single")


@pytest.mark.integration
def test_calc_phi_ase_mpi_openpmd_round_trip_when_mpi_enabled(tmp_path):
    _round_trip_calc_phi_ase(tmp_path, "mpi")


def test_writeInput_preserves_custom_fields(tmp_path):
    try:
        _io()
    except ImportError as exc:
        pytest.skip(str(exc))

    class ThermalPrism(PrismSchema):
        temperature = PrimitiveFieldSpec("temperature", "custom_temperature", np.float64, unit="K", backendRequired=False)

    values = np.array(
        [
            [300.0, 301.0, 302.0, 303.0, 304.0],
            [310.0, 311.0, 312.0, 313.0, 314.0],
            [320.0, 321.0, 322.0, 323.0, 324.0],
        ],
        dtype=np.float64,
    )
    medium = asymmetric_medium().withPrimitiveSchema(ThermalPrism, temperature=values)
    assert next(iter(medium.getPrisms())).temperature == pytest.approx(300.0)
    output = tmp_path / "custom.bp"

    transport.writeInput(output, asymmetric_phi_ase(), medium, asymmetric_cross_sections())

    io = _io()
    series = io.Series(str(output), io.Access.read_only)
    iteration = series.iterations[0]
    record = iteration.meshes["custom_temperature"]
    np.testing.assert_array_equal(_read_scalar(series, iteration, "custom_temperature"), values.reshape(-1, order="F"))
    assert record.get_attribute("haseSchemaVersion") == HASE_SCHEMA_VERSION
    assert record.get_attribute("haseEntity") == "cell_layer"
    assert _attribute_list(record.get_attribute("haseAxes")) == ["cell", "layer"]
    assert record.get_attribute("haseLayoutOrder") == "backendFlat"
    assert _attribute_list(record.get_attribute("hasePrimitiveShape")) == [3, 5]
    assert record.get_attribute("haseStatic") is True
    assert record.get_attribute("haseDynamic") is False
    assert record.get_attribute("haseBackendRequired") is False
    assert record.get_attribute("haseUserDefined") is True
    assert record.get_attribute("haseUserFieldName") == "temperature"
    assert record.get_attribute("haseUnit") == "K"
    assert list(record.axis_labels) == ["flatIndex"]
    series.close()
