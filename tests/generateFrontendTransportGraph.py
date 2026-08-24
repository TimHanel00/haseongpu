#!/usr/bin/env python3

"""Generate the frontend transport graph consumed by the native integration test."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from HASEonGPU import (
    CrossSectionTable,
    Domain,
    ExplicitEuler,
    GainMedium,
    Material,
    OpticalComponent,
    PhiASE,
    Simulation,
    VolumeTopology,
    units,
)
from pyInclude.openpmd.transport import OpenPmdInputSeries


def removePreviousGraph(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def buildSimulation() -> Simulation:
    crossSections = CrossSectionTable.monochromatic(
        wavelength=940 * units.nm,
        absorption=1.2e-21 * units.cm**2,
        emission=2.1e-20 * units.cm**2,
    )
    material = Material(
        materialName="transport integration material",
        temperature=293.15 * units.K,
        refractiveIndex=1.8,
        fluorescenceLifetime=9.5e-4 * units.s,
        crossSections=crossSections,
        active=True,
        activeIonDensity=2.76e20 / units.cm**3,
    )
    topology = VolumeTopology.fromTetrahedra(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        [[0, 1, 2, 3]],
    )
    component = OpticalComponent(
        material=material,
        domain=Domain.fromTopology(topology),
    )
    return Simulation(
        opticalComponents=(component,),
        gainMedium=GainMedium((component,)),
        phiASE=PhiASE(backend="test-backend", ase_steps=0),
        timeIntegrator=ExplicitEuler(),
        timeStepSize=1.0e-6,
        simulationSteps=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("backend", choices=("adios", "hdf5"))
    args = parser.parse_args()

    output = args.output.resolve()
    removePreviousGraph(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with OpenPmdInputSeries(output, backend=args.backend) as writer:
        writer.write(buildSimulation(), iteration_index=0)


if __name__ == "__main__":
    main()
