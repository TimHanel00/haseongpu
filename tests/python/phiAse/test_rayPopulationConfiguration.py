"""Runtime ray-population controls, independent of device scheduling."""

import argparse

import pytest

from pyInclude import PhiASE


def testRayPopulationDefaultsAndCli():
    assert PhiASE().numIndependentRayPopulations == 8
    parser = argparse.ArgumentParser()
    PhiASE.addArguments(parser)
    configured = PhiASE.fromArgs(parser.parse_args(["--num-independent-ray-populations", "12"]))
    assert configured.numIndependentRayPopulations == 12
    assert configured.openPmdAttributes(numberOfSamples=1)["numIndependentRayPopulations"] == 12


def testRayPopulationYaml(tmp_path):
    # This is configuration data, not a regenerated physics reference.
    config = tmp_path / "populations.yaml"
    config.write_text("schema_version: 3\nsimulation:\n  phi_ase:\n    num_independent_ray_populations: 3\n")
    assert PhiASE.fromYaml(config).numIndependentRayPopulations == 3


@pytest.mark.parametrize("count", [0, -1, 1.5, True, float("nan"), float("inf"), 2**32])
def testRayPopulationRejectsInvalidCount(count):
    with pytest.raises(ValueError, match="numIndependentRayPopulations"):
        PhiASE(numIndependentRayPopulations=count).openPmdAttributes(numberOfSamples=1)
