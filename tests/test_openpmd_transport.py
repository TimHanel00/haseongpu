import numpy as np
import pytest

import pyInclude.openpmd_transport as openpmd_transport
from pyInclude.structures import ComputeParameters, ExperimentParameters, HostMesh


pytest.importorskip("pyInclude.openpmd_transport")


def test_write_input_openpmd_contains_frontend_fields(tmp_path):
    mesh = HostMesh(
        trianglePointIndices=np.array([0, 0, 1, 1, 2, 3], dtype=np.uint32),
        numberOfTriangles=2,
        numberOfLevels=3,
        numberOfPoints=4,
        thickness=0.25,
        points=np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        triangleCenterX=np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
        triangleCenterY=np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64),
        triangleNormalPoint=np.array([0, 0, 1, 2, 2, 3], dtype=np.uint32),
        triangleNormalsX=np.array([0.0, 0.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float64),
        triangleNormalsY=np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        forbiddenEdge=np.array([-1, 2, -1, -1, 0, -1], dtype=np.int32),
        triangleNeighbors=np.array([-1, 0, -1, -1, 1, -1], dtype=np.int32),
        triangleSurfaces=np.array([0.5, 0.5], dtype=np.float32),
        betaVolume=np.array([0.1, 0.3, 0.2, 0.4], dtype=np.float64),
        betaCells=np.arange(12, dtype=np.float64),
        claddingCellTypes=np.array([0, 1], dtype=np.uint32),
        refractiveIndices=np.array([1.8, 1.0, 1.8, 1.0], dtype=np.float32),
        reflectivities=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        nTot=5.0,
        crystalTFluo=1.23,
        claddingNumber=2,
        claddingAbsorption=0.05,
    )
    experiment = ExperimentParameters(
        minRaysPerSample=1,
        maxRaysPerSample=2,
        lambdaA=np.array([900.0, 910.0]),
        lambdaE=np.array([1000.0, 1010.0]),
        sigmaA=np.array([0.01, 0.02]),
        sigmaE=np.array([0.03, 0.04]),
        mseThreshold=0.5,
        useReflections=True,
        spectral=2,
    )
    compute = ComputeParameters(
        maxRepetitions=1,
        adaptiveSteps=1,
        maxGpus=1,
        backend="Host_Cpu_CpuSerial",
        parallelMode="single",
    )

    output = tmp_path / "frontend.bp"
    openpmd_transport.write_input(output, experiment, compute, mesh)

    io = openpmd_transport._io()

    series = io.Series(str(output), io.Access.read_only)
    iteration = series.iterations[0]
    assert iteration.get_attribute("number_of_points") == 4
    assert iteration.get_attribute("number_of_cells") == 2
    assert iteration.get_attribute("number_of_levels") == 3
    assert "core_connectivity" in iteration.meshes
    assert "core_beta_volume" in iteration.meshes
    assert "core_point_beta" in iteration.meshes
    assert "core_reflectivity" in iteration.meshes
    series.close()
