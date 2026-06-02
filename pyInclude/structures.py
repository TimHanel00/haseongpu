from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ExperimentParameters:
    minRaysPerSample: int = 100000
    maxRaysPerSample: int = 100000
    lambdaA: Sequence[float] = field(default_factory=list)
    lambdaE: Sequence[float] = field(default_factory=list)
    sigmaA: Sequence[float] = field(default_factory=list)
    sigmaE: Sequence[float] = field(default_factory=list)
    maxSigmaA: float = 0.0
    maxSigmaE: float = 0.0
    mseThreshold: float = 0.1
    useReflections: bool = True
    spectral: int = 0
    monochromatic: bool = False


@dataclass
class ComputeParameters:
    maxRepetitions: int = 4
    adaptiveSteps: int = 4
    maxGpus: int = 1
    gpu_i: int = 0
    backend: str = "gpu"
    parallelMode: str = "single"
    writeVtk: bool = False
    devices: Sequence[int] = field(default_factory=list)
    minSampleRange: int = 0
    maxSampleRange: int = 2**32 - 1
    rngSeed: int | None = None


@dataclass
class HostMesh:
    trianglePointIndices: Sequence[int] = field(default_factory=list)
    numberOfTriangles: int = 0
    numberOfLevels: int = 0
    numberOfPoints: int = 0
    thickness: float = 0.0
    points: Sequence[float] = field(default_factory=list)
    triangleCenterX: Sequence[float] = field(default_factory=list)
    triangleCenterY: Sequence[float] = field(default_factory=list)
    triangleNormalPoint: Sequence[int] = field(default_factory=list)
    triangleNormalsX: Sequence[float] = field(default_factory=list)
    triangleNormalsY: Sequence[float] = field(default_factory=list)
    forbiddenEdge: Sequence[int] = field(default_factory=list)
    triangleNeighbors: Sequence[int] = field(default_factory=list)
    triangleSurfaces: Sequence[float] = field(default_factory=list)
    betaVolume: Sequence[float] = field(default_factory=list)
    betaCells: Sequence[float] = field(default_factory=list)
    claddingCellTypes: Sequence[int] = field(default_factory=list)
    refractiveIndices: Sequence[float] = field(default_factory=list)
    reflectivities: Sequence[float] = field(default_factory=list)
    nTot: float = 0.0
    crystalTFluo: float = 0.0
    claddingNumber: int = 0
    claddingAbsorption: float = 0.0

    def calcTotalReflectionAngles(self):
        return None


@dataclass
class Result:
    phiAse: Sequence[float] = field(default_factory=list)
    mse: Sequence[float] = field(default_factory=list)
    totalRays: Sequence[int] = field(default_factory=list)
    dndtAse: Sequence[float] = field(default_factory=list)
