# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later


import argparse

from HASEonGPU import PhiASE
import pyInclude.simulation as simulation_module

class DummyResult:
    phiAse = [1.0]
    mse = [0.0]
    totalRays = [4]
    dndtAse = [0.0]


def testSimulationRunUsesOpenPmdTransportAndStoresResults(
    monkeypatch,
    smallGainMedium,
    crossSections,
    phiAseTestConfigPath,
):
    captured = {}

    def fakeRunPhiAse(phiAse, gainMedium, spectralProperties):
        captured["phi_ase"] = phiAse
        captured["gain_medium"] = gainMedium
        captured["spectral_properties"] = spectralProperties
        return DummyResult()

    monkeypatch.setattr(simulation_module.transport, "runPhiASE", fakeRunPhiAse)

    phiAse = PhiASE.fromYaml(
        phiAseTestConfigPath,
        spectralProperties=crossSections,
        repetitions=1,
        adaptiveSteps=1,
        backend="Host_Cpu_CpuOmpBlocks",
        parallelMode="single",
        useReflections=False,
        rngSeed=1234,
    ).run(gainMedium=smallGainMedium)

    assert isinstance(phiAse.getResults(), DummyResult)
    assert captured["phi_ase"] is phiAse
    assert captured["gain_medium"] is smallGainMedium
    assert captured["spectral_properties"] is crossSections
    assert captured["phi_ase"].minRaysPerSample == 1000
    assert captured["phi_ase"].useReflections is False
    assert captured["phi_ase"].rngSeed == 1234


def testPhiAseLoadsYamlAndArgumentOverrides(phiAseTestConfigPath):
    phiAse = PhiASE(phiAseTestConfigPath)

    assert phiAse.minRaysPerSample == 1000
    assert phiAse.maxRaysPerSample == 10000
    assert phiAse.repetitions == 1
    assert phiAse.backend == "Host_Cpu_CpuSerial"

    parser = argparse.ArgumentParser()
    PhiASE.addArguments(parser)
    args = parser.parse_args([
        "--phi-ase-config",
        str(phiAseTestConfigPath),
        "--min-rays-per-sample",
        "32",
    ])

    fromArgs = PhiASE.fromArgs(args)

    assert fromArgs.minRaysPerSample == 32
    assert fromArgs.maxRaysPerSample == 10000


def testPhiAseMpiRunUsesOpenPmdTransportMetadata(
    monkeypatch,
    smallGainMedium,
    crossSections,
    phiAseTestConfigPath,
):
    captured = {}

    def fakeRunPhiAse(phiAse, gainMedium, spectralProperties):
        captured["nPerNode"] = phiAse.nPerNode
        captured["numDevices"] = phiAse.numDevices
        captured["parallelMode"] = phiAse.parallelMode
        captured["gain_medium"] = gainMedium
        captured["spectral_properties"] = spectralProperties
        return DummyResult()

    monkeypatch.setattr(simulation_module.transport, "runPhiASE", fakeRunPhiAse)

    phiAse = PhiASE.fromYaml(
        phiAseTestConfigPath,
        spectralProperties=crossSections,
        parallelMode="mpi",
        numDevices=4,
        nPerNode=2,
    ).run(gainMedium=smallGainMedium)

    assert isinstance(phiAse.getResults(), DummyResult)
    assert captured["nPerNode"] == 2
    assert captured["numDevices"] == 4
    assert captured["parallelMode"] == "mpi"
    assert captured["gain_medium"] is smallGainMedium
    assert captured["spectral_properties"] is crossSections


def testPhiAseNPerNodeLoadsFromArgsAndConfig():
    phiAse = PhiASE({"compute": {"nPerNode": 3}})
    assert phiAse.nPerNode == 3

    parser = argparse.ArgumentParser()
    PhiASE.addArguments(parser)
    args = parser.parse_args(["--n-per-node", "5"])

    fromArgs = PhiASE.fromArgs(args)
    assert fromArgs.nPerNode == 5
