#!/usr/bin/env python3
# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import HASEonGPU
import numpy as np

from HASEonGPU import (
    CrossSectionTable,
    Domain,
    GainMedium,
    Material,
    OpticalComponent,
    PhiASE,
    TransportResult,
    VolumeTopology,
)
from hase_units import units


module_path = Path(HASEonGPU.__file__).resolve()
module_path_str = str(module_path)

print("module:", module_path)
assert "site-packages" in module_path_str or "dist-packages" in module_path_str, module_path

topology = VolumeTopology.fromTetrahedra(
    np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ),
    np.asarray([[0, 1, 2, 3]], dtype=np.uint32),
)
material = Material(
    materialName="smoke-test",
    temperature=293.15 * units.K,
    refractiveIndex=1.5,
    fluorescenceLifetime=1.0 * units.ms,
    crossSections=CrossSectionTable.monochromatic(
        wavelength=1030.0 * units.nm,
        absorption=1.0e-21 * units.cm**2,
        emission=2.0e-20 * units.cm**2,
    ),
    activeIonDensity=1.0e20 / units.cm**3,
)
component = OpticalComponent(
    domain=Domain.fromTopology(topology),
    material=material,
)
gain_medium = GainMedium([component])
phi_ase = PhiASE(backend="Host_Cpu_CpuSerial")
result = TransportResult()

print("GainMedium type:", type(gain_medium))
print("PhiASE type:", type(phi_ase))
print("TransportResult type:", type(result))
print("numberOfCells:", topology.numberOfCells)

assert gain_medium.domain.maskFor(topology).tolist() == [True]
assert result.phiAse == []
assert phi_ase.minRays == 100000

for legacy_name in ("HostMesh", "ExperimentParameters", "ComputeParameters", "Mesh"):
    assert not hasattr(HASEonGPU, legacy_name), legacy_name
