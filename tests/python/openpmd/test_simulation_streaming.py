# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyInclude.openpmd import transport


def test_synchronized_debug_exchanges_control_after_each_snapshot(monkeypatch, tmp_path):
    initial_written = threading.Event()
    controls_written = threading.Event()
    writes = []

    class FakeInputSeries:
        def __init__(self, path, *, backend=None):
            assert Path(path).name == "input.sst"
            assert backend == "adios-sst"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def write_simulation(self, simulation, *, iteration_index, run_control):
            writes.append((iteration_index, "initial", None))
            initial_written.set()

        def write_control(self, simulation, *, iteration_index, control_values):
            writes.append((iteration_index, "control", control_values))
            if iteration_index == 2:
                controls_written.set()

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            pass

        def poll(self):
            return None

        def kill(self):
            raise AssertionError("the successful backend must not be killed")

        def communicate(self):
            assert controls_written.wait(timeout=2.0)
            return "", ""

    def fake_read_simulation_output(path, *, on_state=None):
        assert initial_written.wait(timeout=2.0)
        states = [SimpleNamespace(step=step) for step in (1, 2, 3)]
        for state in states:
            on_state(state)
        return states

    monkeypatch.setattr(transport, "OpenPmdInputSeries", FakeInputSeries)
    monkeypatch.setattr(transport.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(transport, "read_simulation_output", fake_read_simulation_output)

    simulation = SimpleNamespace(
        executionMode="synchronized-debug",
        controlFields=("beta_volume",),
    )
    states = transport._run_streaming_simulation(
        ["calcPhiASE", "--cpp-control"],
        tmp_path / "input.sst",
        tmp_path / "output.sst",
        SimpleNamespace(name="adios-sst"),
        simulation,
        {"number_of_steps": 3},
        on_state=lambda state: {"beta_volume": np.asarray([0.1 * state.step])},
    )

    assert states[-1].step == 3
    assert writes[0] == (0, "initial", None)
    assert writes[1][0:2] == (1, "control")
    assert writes[2][0:2] == (2, "control")
    np.testing.assert_allclose(writes[1][2]["beta_volume"], [0.1])
    np.testing.assert_allclose(writes[2][2]["beta_volume"], [0.2])
