# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Construct a simulation from a resolved material condition."""

from _source_tree_import import ensure_hase_importable

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
    Domain,
    ExplicitEuler,
    GainMedium,
    Material,
    OpticalComponent,
    PhiASE,
    Simulation,
    SurfaceOptics,
    VolumeTopology,
    units,
)
from material_library import loadBuiltinMaterials  # noqa: E402


def buildSimulation():
    """Build a two-cell gain/cladding assembly from material-bound domains.

    The active and passive components partition one shared topology. The
    simulation receives an explicitly constructed exterior surface, while a
    reflective coating is assigned independently to one cladding face.
    """
    gainMaterial = loadBuiltinMaterials().resolve(
        "YbYAG",
        temperature=293.15 * units.K,
        activeIonDensity=2.776e20 / units.cm**3,
    )
    topology = VolumeTopology.fromTetrahedra(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        cellPointIndices=[[0, 1, 2, 3], [1, 2, 3, 4]],
        cellDomains=[1, 2],
        metadata={"cellDomainNames": {1: "gain", 2: "cladding"}},
    )
    gainDomain = Domain.fromGmsh(topology, "gain", entityKind="volume")
    claddingDomain = Domain.fromGmsh(topology, "cladding", entityKind="volume")
    crystal = OpticalComponent(
        domain=gainDomain,
        material=gainMaterial,
        name="crystal",
    )
    cladding = OpticalComponent(
        domain=claddingDomain,
        material=Material(
            materialName="absorbing cladding",
            temperature=293.15 * units.K,
            refractiveIndex=1.45,
            fluorescenceLifetime=None,
            crossSections=None,
            active=False,
            bulkAttenuation=5.5 / units.cm,
        ),
        name="cladding",
    )
    mirrorSurface = Domain.where(topology, "x_max")
    cladding.assignSurfaceOptics(
        mirrorSurface,
        SurfaceOptics(reflectivity=0.98, n_inside=1.45, n_outside=1.0),
    )
    gainMedium = GainMedium([crystal], name="amplifier")
    return Simulation(
        opticalComponents=[crystal, cladding],
        gainMedium=gainMedium,
        exteriorSurface=(gainDomain + claddingDomain).boundary(),
        initialExcitation=0.0,
        phiASE=PhiASE(ase_steps=0),
        timeIntegrator=ExplicitEuler(),
        timeStepSize=1.0e-6,
        simulationSteps=1,
    )
