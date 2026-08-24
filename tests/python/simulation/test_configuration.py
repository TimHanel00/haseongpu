# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import pytest
import yaml

from HASEonGPU import (
    CrossSectionTable,
    ImplicitEuler,
    Material,
    Simulation,
    SuperGaussianPumpProfile,
    units,
)


def _material():
    return Material(
        materialName="test gain material",
        temperature=293.15 * units.K,
        refractiveIndex=1.8,
        fluorescenceLifetime=9.5e-4 * units.s,
        crossSections=CrossSectionTable.monochromatic(
            wavelength=940 * units.nm,
            absorption=1.2e-21 * units.cm**2,
            emission=2.1e-20 * units.cm**2,
        ),
        active=True,
        activeIonDensity=2.76e20 / units.cm**3,
    )


def _config():
    return {
        "schema_version": 3,
        "topologies": {
            "crystal_mesh": {
                "from_tetrahedra": {
                    "points": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "cell_point_indices": [[0, 1, 2, 3]],
                }
            }
        },
        "domains": {
            "crystal_volume": {"topology": "crystal_mesh"},
            "pump_face": {
                "topology": {"name": "crystal_mesh", "entity_kind": "surface"}
            },
        },
        "optical_components": {
            "crystal": {
                "domain": "crystal_volume",
                "material": "gain_material",
                "surface_optics": [
                    {
                        "domain": "pump_face",
                        "reflectivity": 0.75,
                        "exterior_refractive_index": 1.0,
                    }
                ],
            }
        },
        "gain_media": {"amplifier": {"components": ["crystal"]}},
        "simulation": {
            "optical_components": ["crystal"],
            "gain_medium": "amplifier",
            "exterior_surface": "pump_face",
            "initial_excitation": {"value": 0.0},
            "phi_ase": {
                "propagation_mode": "forward",
                "min_rays": 10,
                "max_rays": 100,
                "forward_ray_count": 12,
                "relative_standard_error_threshold": 0.05,
                "repetitions": 2,
                "adaptive_steps": 3,
                "use_reflections": True,
                "reflection_max_iterations": 7,
                "reflection_tolerance": 1.0e-5,
                "backend": "Host_Cpu_CpuSerial",
                "ase_steps": 3,
            },
            "pumps": [
                {
                    "name": "main",
                    "total_power": 10.0,
                    "ray_count": 1234,
                    "pump_steps": 2,
                    "rng_seed": 99,
                    "spectrum": {"monochromatic": 940e-9},
                    "angular_distribution": {
                        "uniform_cone": {
                            "half_angle": 0.1,
                            "polar_samples": 2,
                            "azimuthal_samples": 3,
                        }
                    },
                    "profile": {
                        "kind": "super_gaussian",
                        "radius_u": 1.5,
                        "radius_v": 1.25,
                        "exponent": 40.0,
                    },
                    "injection": {"domain": "pump_face"},
                }
            ],
            "time_integrator": {
                "method": "implicit_euler",
                "iterations": 3,
                "tolerance": 1.0e-8,
            },
            "time_step_size": 2.0e-5,
            "simulation_steps": 3,
            "pre_pump": True,
            "report_timings": True,
            "output_steps": [2, 3],
            "output_fields": ["beta_volume", "phi_ase", "dndt_pump"],
        },
    }


def _write(path: Path, config):
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _load(tmp_path, config=None):
    return Simulation.fromYaml(
        _write(tmp_path / "simulation.yaml", _config() if config is None else config),
        materials={"gain_material": _material()},
    )


def testSimulationFromYamlBuildsPublicObjectGraph(tmp_path):
    simulation = _load(tmp_path)

    assert simulation.simulationSteps == 3
    assert simulation.exteriorSurface.entityKind == "surface"
    assert isinstance(simulation.timeIntegrationSolver, ImplicitEuler)
    assert simulation.timeIntegrationSolver.iterations == 3
    assert simulation.phiASE.useReflections is True
    assert simulation.phiASE.forwardRayCount == 12
    assert simulation.prePump is True
    assert simulation.gainMedium.components[0].surfaceOptics[0].optics.reflectivity == pytest.approx(0.75)
    assert len(simulation.pumps) == 1
    assert simulation.pumps[0].name == "main"
    assert simulation.pumps[0].ray_count == 1234
    assert isinstance(simulation.pumps[0].profile, SuperGaussianPumpProfile)
    assert simulation.outputSteps == (2, 3)
    assert simulation.outputFields == ("beta_volume", "phi_ase", "dndt_pump")


def testSchemaV3DerivesRunLimitWhenSimulationStepsIsOmitted(tmp_path):
    config = _config()
    config["simulation"].pop("simulation_steps")
    assert _load(tmp_path, config)._derived_simulation_steps() == 3


def testSchemaV3RejectsTwoRunLimits(tmp_path):
    config = _config()
    config["simulation"]["max_time"] = 1.0
    with pytest.raises(ValueError, match="at most one"):
        _load(tmp_path, config)


def testSchemaV3RejectsRemovedCrossSectionRegistry(tmp_path):
    config = _config()
    config["cross_sections"] = {}
    with pytest.raises(ValueError, match="unsupported top-level options"):
        _load(tmp_path, config)


def testSchemaV3RejectsUnknownSurfaceOpticsOptions(tmp_path):
    config = _config()
    config["optical_components"]["crystal"]["surface_optics"][0]["n_inside"] = 1.8
    with pytest.raises(ValueError, match="surface_optics"):
        _load(tmp_path, config)


def testSchemaV3RejectsUnknownReferences(tmp_path):
    config = _config()
    config["gain_media"]["amplifier"]["components"] = ["missing"]
    with pytest.raises(ValueError, match="unknown optical_components reference"):
        _load(tmp_path, config)
