# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from HASEonGPU import AlpakaBackends

repoRoot = Path(__file__).resolve().parents[3]


from pyInclude.geometry.vtk import _parseVtk
from pyInclude.openpmd import transport


_convert_vtk_topology_path = repoRoot / "utils" / "convert_vtk_topology.py"
_convert_vtk_topology_spec = importlib.util.spec_from_file_location(
    "_hase_convert_vtk_topology",
    _convert_vtk_topology_path,
)
if _convert_vtk_topology_spec is None or _convert_vtk_topology_spec.loader is None:
    raise ImportError(f"cannot load VTK topology converter from {_convert_vtk_topology_path}")
_convert_vtk_topology = importlib.util.module_from_spec(_convert_vtk_topology_spec)
sys.modules[_convert_vtk_topology_spec.name] = _convert_vtk_topology
_convert_vtk_topology_spec.loader.exec_module(_convert_vtk_topology)
convertVtk = _convert_vtk_topology.convertVtk


_laser_pump_launcher_path = repoRoot / "utils" / "testLaunchLaserPump.py"
_laser_pump_launcher_spec = importlib.util.spec_from_file_location(
    "_hase_test_launch_laser_pump",
    _laser_pump_launcher_path,
)
if _laser_pump_launcher_spec is None or _laser_pump_launcher_spec.loader is None:
    raise ImportError(f"cannot load laser-pump launcher from {_laser_pump_launcher_path}")
_laser_pump_launcher = importlib.util.module_from_spec(_laser_pump_launcher_spec)
_laser_pump_launcher_spec.loader.exec_module(_laser_pump_launcher)


exampleDir = repoRoot / "example"
sys.path.insert(0, str(exampleDir))
import laserPumpCladding  # noqa: E402


REFERENCE_PATH = (
    repoRoot
    / "tests"
    / "data"
    / "laserPumpCladding"
    / "fixed_legacy_reflection_pump3_wedge_reference"
    / "phiase_reference.npz"
)
# The fixed legacy wedge and Tet4 solvers both reflect at the physical top
# plane, z = (numberOfLevels - 1) * thickness.
INTEGRAL_RTOL = 0.05


def _triangle_area(points):
    a, b, c = np.asarray(points, dtype=np.float64)
    ab = b[:2] - a[:2]
    ac = c[:2] - a[:2]
    return 0.5 * abs(float(ab[0] * ac[1] - ab[1] * ac[0]))


def _wedge_volume(points, cell):
    vertices = np.asarray(points, dtype=np.float64)[np.asarray(cell, dtype=np.uint32)]
    z_values = np.unique(vertices[:, 2])
    if z_values.size != 2:
        raise ValueError("wedge cell must span exactly two z levels")
    lower = vertices[np.isclose(vertices[:, 2], z_values[0])]
    if lower.shape[0] != 3:
        raise ValueError("wedge cell must have three lower vertices")
    return _triangle_area(lower) * float(z_values[1] - z_values[0])


def _tet_volume(points, cell):
    a, b, c, d = np.asarray(points, dtype=np.float64)[np.asarray(cell, dtype=np.uint32)]
    return abs(float(np.dot(b - a, np.cross(c - a, d - a)))) / 6.0


def _tet_cell_integral(points, cells, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    cells = np.asarray(cells, dtype=np.uint32)
    volumes = np.asarray([_tet_volume(points, cell) for cell in cells], dtype=np.float64)
    if values.shape != volumes.shape:
        raise ValueError(f"Tet4 field has {values.size} values for {volumes.size} cells")
    return float(np.sum(values * volumes))


def _wedge_point_integrals(points, cells, phi_by_step):
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.uint32)
    volumes = np.asarray([_wedge_volume(points, cell) for cell in cells], dtype=np.float64)
    result = []
    for phi in np.asarray(phi_by_step, dtype=np.float64):
        cell_values = phi[cells].mean(axis=1)
        result.append(float(np.sum(cell_values * volumes)))
    return np.asarray(result, dtype=np.float64)


def _vtkScalarNames(path):
    tokens = path.read_text(encoding="utf-8").split()
    return {tokens[index + 1] for index, token in enumerate(tokens) if token.upper() == "SCALARS"}


@pytest.fixture(scope="module")
def laserPumpCladdingReference():
    if not REFERENCE_PATH.is_file():
        pytest.skip(f"missing generated laserPumpCladding reference data: {REFERENCE_PATH}")
    with np.load(REFERENCE_PATH, allow_pickle=False) as data:
        return {
            "metadata": json.loads(str(data["metadata"].item())),
            "phiASE": np.asarray(data["phiASE"], dtype=np.float64),
            "points": np.asarray(data["points"], dtype=np.float64),
            "cells": np.asarray(data["cells"], dtype=np.uint32),
            "cellTypes": np.asarray(data["cellTypes"], dtype=np.uint32),
        }


def testLaserPumpCladdingReferenceDocumentsGeneratorAndSeed(laserPumpCladdingReference):
    metadata = laserPumpCladdingReference["metadata"]

    assert metadata["generator"]["commit"] == "effd8077edccef93a68d818e8a5eb2f0ebdc03b4"
    assert metadata["parameters"]["backend"] == "Host_Cpu_CpuOmpBlocks"
    assert metadata["parameters"]["timeSlices"] == 6
    assert metadata["parameters"]["pumpSteps"] == 3
    assert metadata["random"]["rngSeed"] == 5489
    assert metadata["observable"]["field"] == "phiASE"
    assert metadata["observable"]["location"] == "legacy wedge POINT_DATA"
    assert metadata["observable"]["coverage"] == "full point buffer"


def testLaserPumpCladdingReferenceStoresFullWedgePointBuffers(laserPumpCladdingReference):
    metadata = laserPumpCladdingReference["metadata"]
    phi = laserPumpCladdingReference["phiASE"]

    assert phi.shape == tuple(metadata["observable"]["shape"])
    assert phi.shape == (6, 4210)
    assert laserPumpCladdingReference["points"].shape == (4210, 3)
    assert laserPumpCladdingReference["cells"].shape == (7308, 6)
    assert np.all(laserPumpCladdingReference["cellTypes"] == 13)
    assert np.isfinite(phi).all()
    np.testing.assert_allclose(phi.mean(axis=1), metadata["meanPhi"], rtol=0.0, atol=0.0)


def testPtTet4GeometryConvertsBackToLegacyWedgeOrder(
    laserPumpCladdingReference,
    tmp_path,
):
    wedge_path = convertVtk(
        repoRoot / "example" / "data" / "ptTet4.vtk",
        tmp_path / "pt_wedge.vtk",
        direction="tet4-to-wedge",
    )
    points, cells, cell_types, point_data, cell_data, _fields = _parseVtk(wedge_path)

    np.testing.assert_allclose(points, laserPumpCladdingReference["points"], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        np.sort(np.asarray(cells, dtype=np.uint32), axis=1),
        np.sort(laserPumpCladdingReference["cells"], axis=1),
    )
    np.testing.assert_array_equal(np.asarray(cell_types, dtype=np.uint32), laserPumpCladdingReference["cellTypes"])
    assert "betaCells" in point_data
    assert "betaVolume" in cell_data


def testLaserPumpCladdingMaterialKeepsRawSpectrumAtItsNativeResolution():
    spectra = laserPumpCladding.laser_pump_cladding_material(191).crossSections
    material_dir = repoRoot / "material_library" / "data" / "legacy_yb_yag"
    raw_wavelengths_absorption = np.loadtxt(material_dir / "lambda_a.txt")
    raw_absorption = np.loadtxt(material_dir / "sigma_a.txt")
    raw_wavelengths_emission = np.loadtxt(material_dir / "lambda_e.txt")
    raw_emission = np.loadtxt(material_dir / "sigma_e.txt")

    assert raw_wavelengths_absorption.size == 191
    np.testing.assert_array_equal(spectra.wavelengths.toValue(laserPumpCladding.units.nm), raw_wavelengths_absorption)
    np.testing.assert_array_equal(spectra.wavelengths.toValue(laserPumpCladding.units.nm), raw_wavelengths_emission)
    np.testing.assert_array_equal(spectra.absorption.toValue(laserPumpCladding.units.cm**2), raw_absorption)
    np.testing.assert_array_equal(spectra.emission.toValue(laserPumpCladding.units.cm**2), raw_emission)


def testLaserPumpCladdingMaterialResamplesSpectrumBeforeTransport():
    material = laserPumpCladding.laser_pump_cladding_material(1000)

    assert material.crossSections.wavelengths.size == 1000
    assert material.crossSections.metadata["spectral_resampling"] == {
        "method": "linear",
        "source_sample_count": 191,
        "sample_count": 1000,
    }


def testLaserPumpCladdingTet4MeshAssignsCylinderSurfaceDomains():
    topology = laserPumpCladding.laser_pump_cladding_mesh()
    boundaries = np.asarray(topology.faceBoundaries, dtype=np.int32)
    points = np.asarray(topology.points, dtype=np.float64)
    face_nodes = np.asarray(topology.facePointIndices, dtype=np.uint32)
    face_z = points[:, 2][face_nodes]

    bottom_id = laserPumpCladding.BOTTOM_ASE_SURFACE_ID
    top_id = laserPumpCladding.TOP_ASE_SURFACE_ID

    cladding_id = laserPumpCladding.CLADDING_SURFACE_ID

    assert topology.surfaceDomainNames == {
        bottom_id: "ase_bottom",
        top_id: "ase_top",
        cladding_id: "cladding",
    }
    assert np.count_nonzero(boundaries == bottom_id) == 812
    assert np.count_nonzero(boundaries == top_id) == 812
    assert np.count_nonzero(boundaries == cladding_id) == 432
    assert np.count_nonzero(topology.neighborCells < 0) == 2056
    np.testing.assert_allclose(face_z[boundaries == bottom_id], np.min(points[:, 2]), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(face_z[boundaries == top_id], np.max(points[:, 2]), rtol=0.0, atol=1.0e-12)



def testLaserPumpCladdingTet4MeshPreservesTenLayerPumpLayout():
    topology = laserPumpCladding.laser_pump_cladding_mesh()
    points = np.asarray(topology.points, dtype=np.float64)
    z_planes = np.unique(points[:, 2])

    assert topology.cellDomainNames == {1: "crystal_volume"}
    np.testing.assert_array_equal(topology.cellDomains, np.ones(topology.numberOfCells, dtype=np.int32))
    assert z_planes.size == laserPumpCladding.NUMBER_OF_Z_LAYERS
    np.testing.assert_allclose(
        z_planes,
        np.linspace(0.0, 0.7, laserPumpCladding.NUMBER_OF_Z_LAYERS),
        rtol=0.0,
        atol=1.0e-12,
    )

    assert topology.structuredNumberOfLevels == laserPumpCladding.NUMBER_OF_Z_LAYERS
    assert topology.numberOfSamplePoints == topology.numberOfPoints
    assert topology.numberOfSamplePoints == topology.structuredNumberOfPoints * topology.structuredNumberOfLevels

    points_by_level = points.reshape(
        (topology.structuredNumberOfLevels, topology.structuredNumberOfPoints, 3)
    )
    np.testing.assert_allclose(
        points_by_level[:, :, 2],
        np.broadcast_to(z_planes[:, None], points_by_level[:, :, 2].shape),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        points_by_level[:, :, :2],
        np.broadcast_to(points_by_level[0, :, :2], points_by_level[:, :, :2].shape),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.integration
def testLaserPumpCladdingRunExampleReflectionToggleChangesPhiAse(
    tmp_path,
    alpakaRuntimeBackend,
    openPmdRuntimeBackend,
):
    def run(use_reflections):
        output_dir = tmp_path / f"reflections_{int(use_reflections)}"
        laserPumpCladding.run_example(
            backend=alpakaRuntimeBackend,
            openPmdBackend=openPmdRuntimeBackend,
            timeSteps=2,
            pumpSteps=1,
            vtkOutputDir=output_dir,
            enableAse=True,
            prePump=True,
            rngSeed=1234,
            useReflections=use_reflections,
            minRays=20_000,
            maxRays=20_000,
            adaptiveSteps=1,
            relativeStandardErrorThreshold=0.1,
            reflection_max_iterations=17,
            reflection_tolerance=0.0,
            surface_reservoir_size=32,
        )
        points, cells, _cell_types, _point_data, cell_data, _fields = _parseVtk(
            output_dir / "laserPumpCladding_002.vtk"
        )
        return _tet_cell_integral(points, cells, cell_data["volumePhiASE"])

    without_reflections = run(False)
    with_reflections = run(True)

    assert without_reflections > 0.0
    assert with_reflections > without_reflections * 1.05


@pytest.fixture(scope="module")
def laserPumpCladdingBackendResults(
    laserPumpCladdingReference,
    openPmdRuntimeBackend,
    tmp_path_factory,
):
    metadata = laserPumpCladdingReference["metadata"]
    results = {}
    for alpakaBackend in AlpakaBackends.all():
        tet4_dir = tmp_path_factory.mktemp(f"laser_pump_{openPmdRuntimeBackend}_{alpakaBackend}")
        state = laserPumpCladding.run_example(
            backend=alpakaBackend,
            openPmdBackend=openPmdRuntimeBackend,
            timeSteps=metadata["parameters"]["timeSlices"],
            pumpSteps=metadata["parameters"]["pumpSteps"],
            vtkOutputDir=tet4_dir,
            enableAse=True,
            prePump=metadata["parameters"]["prePump"],
            rngSeed=metadata["random"]["rngSeed"],
            useReflections=metadata["parameters"]["useReflections"],
            minRays=metadata["parameters"]["minRaysPerSample"],
            maxRays=metadata["parameters"]["maxRaysPerSample"],
            relativeStandardErrorThreshold=0.05,
            adaptiveSteps=metadata["parameters"]["adaptiveSteps"],
        )

        relative_standard_error = np.asarray(state.volumeRelativeStandardError, dtype=np.float64)
        defined_relative_standard_error = relative_standard_error[np.isfinite(relative_standard_error)]
        observed_integrals = []
        for step in metadata["observable"]["stepNumbers"]:
            tet4_path = tet4_dir / f"laserPumpCladding_{step:03d}.vtk"
            tet4_points, tet4_cells, _tet4_cell_types, _tet4_point_data, tet4_cell_data, _tet4_fields = _parseVtk(
                tet4_path
            )
            observed_integrals.append(
                _tet_cell_integral(tet4_points, tet4_cells, tet4_cell_data["volumePhiASE"])
            )
        results[alpakaBackend] = {
            "integrals": np.asarray(observed_integrals, dtype=np.float64),
            "relativeStandardError": defined_relative_standard_error,
        }
    return results


@pytest.mark.integration
def testCurrentTet4ForwardPhiAseMatchesLegacyWedgeReferenceIntegral(
    laserPumpCladdingReference,
    laserPumpCladdingBackendResults,
):
    for alpakaBackend, result in laserPumpCladdingBackendResults.items():
        relative_standard_error = result["relativeStandardError"]
        assert relative_standard_error.size > 0, alpakaBackend
        assert np.max(relative_standard_error) < 0.15, alpakaBackend

    reference_integrals = _wedge_point_integrals(
        laserPumpCladdingReference["points"],
        laserPumpCladdingReference["cells"],
        laserPumpCladdingReference["phiASE"],
    )
    for alpakaBackend, result in laserPumpCladdingBackendResults.items():
        np.testing.assert_allclose(
            result["integrals"],
            reference_integrals,
            rtol=INTEGRAL_RTOL,
            atol=0.0,
            err_msg=alpakaBackend,
        )


@pytest.mark.integration
def testLaserPumpCladdingPhiAseIntegralsAgreeAcrossAlpakaBackends(laserPumpCladdingBackendResults):
    for (left_backend, left_result), (right_backend, right_result) in itertools.combinations(
        laserPumpCladdingBackendResults.items(),
        2,
    ):
        left = left_result["integrals"]
        right = right_result["integrals"]
        scale = np.maximum(np.abs(left), np.abs(right))
        relative_difference = np.divide(
            np.abs(left - right),
            scale,
            out=np.zeros_like(scale),
            where=scale > 0.0,
        )
        np.testing.assert_array_less(
            relative_difference,
            INTEGRAL_RTOL,
            err_msg=f"{left_backend} versus {right_backend}",
        )


def testLaserPumpCladdingUsesPublicMaterialProperties():
    material = laserPumpCladding.laser_pump_cladding_material()

    assert material.activeIonDensity.toValue(laserPumpCladding.units.m**-3) == pytest.approx(2 * 1.388e26)
    assert material.refractiveIndex == 1.83
    assert material.fluorescenceLifetime.toValue(laserPumpCladding.units.s) == pytest.approx(9.41e-4)


@pytest.fixture
def fakeCompiledSnapshots(monkeypatch):
    calls = []

    def fake_run_simulation(
        simulation,
        *,
        steps,
        pump_steps=None,
        transport=None,
        command_prefix=None,
        workspace_dir=None,
    ):
        calls.append(
            {
                "simulation": simulation,
                "steps": steps,
                "pump_steps": pump_steps,
                "transport": transport,
                "command_prefix": command_prefix,
                "workspace_dir": workspace_dir,
                "ase_solver": simulation.aseSolver,
            }
        )
        point_shape = (simulation.mesh.numberOfPoints,)
        volume_shape = (simulation.mesh.numberOfCells,)
        states = []
        for step in range(1, steps + 1):
            pump_active = pump_steps is None or step <= pump_steps
            states.append(
                SimpleNamespace(
                    step=step,
                    time=step * float(simulation.timeStepSize.toValue(laserPumpCladding.units.s)),
                    betaCells=np.full(point_shape, 0.05 * step, dtype=np.float64),
                    betaVolume=np.full(volume_shape, 0.025 * step, dtype=np.float64),
                    phiAse=np.ones(point_shape, dtype=np.float64),
                    dndtAse=np.zeros(point_shape, dtype=np.float64),
                    dndtPump=(
                        np.ones(point_shape, dtype=np.float64)
                        if pump_active
                        else np.zeros(point_shape, dtype=np.float64)
                    ),
                    aseResult=object(),
                )
            )
        return states

    monkeypatch.setattr(transport, "run_simulation", fake_run_simulation)
    return calls


def testLaserPumpCladdingExampleWritesVtkFromCompiledSnapshots(
    tmp_path,
    fakeCompiledSnapshots,
):
    state = laserPumpCladding.run_example(
        timeSteps=2,
        pumpSteps=1,
        vtkOutputDir=tmp_path,
    )

    first = tmp_path / "laserPumpCladding_001.vtk"
    second = tmp_path / "laserPumpCladding_002.vtk"
    assert first.is_file()
    assert second.is_file()
    assert state.step == 2
    scalars = _vtkScalarNames(second)
    assert {"betaCells", "phiASE", "dndtAse", "dndtPump", "cladAbs"}.issubset(scalars)
    assert fakeCompiledSnapshots[-1]["ase_solver"].relativeStandardErrorThreshold == 0.1
    assert fakeCompiledSnapshots[-1]["pump_steps"] == 1
    simulation = fakeCompiledSnapshots[-1]["simulation"]
    assert simulation.resolvedProblem.materials[0].crossSections.wavelengths.size == 1000
    assert "spectralResolution" not in transport._compiled_attribute_values(simulation)


def testLaserPumpCladdingExampleWiresOpenPmdBackend(
    tmp_path,
    fakeCompiledSnapshots,
):
    state = laserPumpCladding.run_example(
        timeSteps=2,
        pumpSteps=1,
        vtkOutputDir=tmp_path,
        openPmdBackend="hdf5",
    )

    assert state.step == 2
    assert fakeCompiledSnapshots[-1]["transport"] == "hdf5"
    assert fakeCompiledSnapshots[-1]["ase_solver"].openPmdBackend == "hdf5"
    assert np.allclose(state.dExcitationDtPump, 0.0)


def testLaserPumpCladdingExampleCanDisableAse(
    tmp_path,
    fakeCompiledSnapshots,
):
    state = laserPumpCladding.run_example(
        timeSteps=1,
        pumpSteps=1,
        vtkOutputDir=tmp_path,
        enableAse=False,
    )

    assert state.step == 1
    assert fakeCompiledSnapshots[-1]["simulation"].enableAse is False


def testLaserPumpCladdingCliAcceptsDisableAse(monkeypatch, tmp_path):
    calls = []

    def fake_run_example(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(
            phiAse=np.zeros((2, 3)),
            sampledExcitationFraction=np.zeros((2, 3)),
        )

    monkeypatch.setattr(laserPumpCladding, "run_example", fake_run_example)

    laserPumpCladding.main(
        [
            "--disable-ase",
            "--time-steps",
            "1",
            "--pump-steps",
            "1",
            "--vtk-output-dir",
            str(tmp_path),
            "--spectral-resolution",
            "191",
        ]
    )

    assert calls[-1]["kwargs"]["enableAse"] is False
    assert calls[-1]["kwargs"]["spectralResolution"] == 191


def testLaserPumpCladdingLauncherUsesSupportedCliOptions(monkeypatch, tmp_path):
    calls = []
    mpi_config = tmp_path / "hase-phiase-mpi.yaml"
    monkeypatch.setenv("HASE_PHIASE_CONFIG", str(mpi_config))

    def fake_run_example(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(
            phiAse=np.zeros((2, 3)),
            sampledExcitationFraction=np.zeros((2, 3)),
        )

    monkeypatch.setattr(laserPumpCladding, "run_example", fake_run_example)
    command = _laser_pump_launcher.launch_command("hdf5", tmp_path)

    laserPumpCladding.main(command[2:])

    assert calls[-1]["kwargs"]["openPmdBackend"] == "hdf5"
    assert calls[-1]["kwargs"]["rngSeed"] == 5489
    assert calls[-1]["kwargs"]["phiAseConfigPath"] == mpi_config


@pytest.mark.parametrize("option", ("--min-sample-range", "--max-sample-range"))
def testLaserPumpCladdingCliRejectsDeprecatedSampleRangeOptions(option):
    with pytest.raises(SystemExit) as exc_info:
        laserPumpCladding.main([option, "0"])

    assert exc_info.value.code == 2
