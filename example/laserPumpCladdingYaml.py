# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run laserPumpCladding with all model and run parameters supplied by YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_tree_import import ensure_hase_importable


scriptDir = Path(__file__).resolve().parent
defaultConfigPath = scriptDir.parent / "config" / "laserPumpCladding.yaml"

ensure_hase_importable()

from HASEonGPU import Simulation, units  # noqa: E402
from laserPumpCladdingApi import printState, writeVtkFields  # noqa: E402
from pyInclude.openpmd.paraview import writeParaviewState  # noqa: E402


def buildSimulation(configPath=defaultConfigPath):
    """Construct the complete example without executing it."""
    return Simulation.fromYaml(configPath)


def runExample(
    configPath=defaultConfigPath,
    vtkOutputDir=scriptDir,
    openPmdOutputDir=None,
):
    """Run the YAML-defined simulation and return its final state."""
    simulation = buildSimulation(configPath)
    material = simulation.gainMedium.components[0].material
    passiveComponents = tuple(
        component
        for component in simulation.opticalComponents
        if component not in simulation.gainMedium.components
    )
    claddingMask = (
        simulation.cellMask(passiveComponents[0].domain)
        if passiveComponents
        else np.zeros(simulation._simulationState.topology.numberOfCells, dtype=bool)
    )
    bulkAttenuation = (
        0.0
        if not passiveComponents or passiveComponents[0].material.bulkAttenuation is None
        else passiveComponents[0].material.bulkAttenuation.toValue(units.cm**-1)
    )

    print(f"Running simulation with backend {simulation.phiASE.backend}")
    print(f"Using openPMD backend {simulation.phiASE.openpmdBackend}")
    simulation.onStep(printState)
    simulation.onStep(
        writeVtkFields,
        Path(vtkOutputDir),
        bulkAttenuation,
        claddingMask,
        material.crossSections,
        material.activeIonDensity.toValue(units.cm**-3),
    )
    if openPmdOutputDir is not None:
        simulation.onStep(
            writeParaviewState,
            Path(openPmdOutputDir),
            bulkAttenuation,
            claddingMask,
        )
    simulation.step()
    return simulation.getLastState()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "YAML-driven HASEonGPU laser-pump cladding example. Edit the YAML "
            "for model, solver, backend, and run parameters."
        )
    )
    parser.add_argument("--config", type=Path, default=defaultConfigPath)
    parser.add_argument("--vtk-output-dir", type=Path, default=scriptDir)
    parser.add_argument("--openpmd-output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    state = runExample(
        args.config,
        vtkOutputDir=args.vtk_output_dir,
        openPmdOutputDir=args.openpmd_output_dir,
    )
    print(f"phiAse shape: {state.phiAse.shape}")
    print(f"betaVolume shape: {state.betaVolume.shape}")


if __name__ == "__main__":
    main()
