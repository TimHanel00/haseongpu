import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _source_backends_module():
    root = Path(__file__).resolve().parents[3]
    names = ("pyInclude", "pyInclude.openpmd", "pyInclude._runtime", "pyInclude.openpmd.backends")
    previous = {name: sys.modules.get(name) for name in names}
    package = types.ModuleType("pyInclude")
    package.__path__ = [str(root / "pyInclude")]
    openpmd_package = types.ModuleType("pyInclude.openpmd")
    openpmd_package.__path__ = [str(root / "pyInclude" / "openpmd")]
    try:
        sys.modules["pyInclude"] = package
        sys.modules["pyInclude.openpmd"] = openpmd_package
        runtime_path = root / "pyInclude" / "_runtime.py"
        runtime_spec = importlib.util.spec_from_file_location("pyInclude._runtime", runtime_path)
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        sys.modules["pyInclude._runtime"] = runtime_module
        runtime_spec.loader.exec_module(runtime_module)
        path = root / "pyInclude" / "openpmd" / "backends.py"
        spec = importlib.util.spec_from_file_location("pyInclude.openpmd.backends", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["pyInclude.openpmd.backends"] = module
        spec.loader.exec_module(module)
        return module, runtime_module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


backends, runtime = _source_backends_module()


class FakeProbeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeProbeLibrary:
    def __init__(self, names):
        self.names = tuple(name.encode("utf-8") for name in names)
        self.haseOpenPmdBackendCount = FakeProbeFunction(lambda: len(self.names))
        self.haseOpenPmdBackendName = FakeProbeFunction(
            lambda index: self.names[index] if index < len(self.names) else None
        )


def test_openPmdBackendsLoadsCompiledProbe(monkeypatch, tmp_path):
    probe_path = tmp_path / "libHaseOpenPmdBackendProbe.so"
    monkeypatch.setattr(
        backends,
        "_load_probe_library",
        lambda extra_dirs=(): (FakeProbeLibrary(("hdf5", "adios", "unsupported")), probe_path),
    )

    assert backends._load_backend_names() == (("adios", "hdf5"), probe_path)


def test_openPmdBackendsCachesNames(monkeypatch):
    calls = []

    def fake_load_backend_names(extra_dirs=()):
        calls.append(tuple(extra_dirs))
        return ("adios",), "probe"

    monkeypatch.setattr(backends, "_load_backend_names", fake_load_backend_names)
    monkeypatch.setattr(backends.OpenPmdBackends, "_known", None)
    monkeypatch.setattr(backends.OpenPmdBackends, "_probe_path", None)

    assert backends.OpenPmdBackends.all() == ["adios"]
    assert backends.OpenPmdBackends.known() == ["adios"]
    assert calls == [()]


def test_openPmdBackendsFailsWhenProbeReportsNoKnownBackends(monkeypatch, tmp_path):
    probe_path = tmp_path / "libHaseOpenPmdBackendProbe.so"
    monkeypatch.setattr(
        backends,
        "_load_probe_library",
        lambda extra_dirs=(): (FakeProbeLibrary(("unsupported",)), probe_path),
    )

    with pytest.raises(RuntimeError, match="did not report any supported backends"):
        backends._load_backend_names()


def test_openPmdProbeCandidatesUseConfiguredRuntimeLib(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    probe_path = runtime / "lib" / "libHaseOpenPmdBackendProbe.so"
    monkeypatch.setenv("HASE_RUNTIME_DIR", str(runtime))

    assert next(backends._candidate_paths()) == probe_path


def testConfiguredOpenPmdPythonPackageIsActivatedBeforeImport(monkeypatch, tmp_path):
    provider = tmp_path / "site-packages"
    (provider / "openpmd_api").mkdir(parents=True)
    monkeypatch.delitem(sys.modules, "openpmd_api", raising=False)
    monkeypatch.setattr(runtime, "_configured_openpmd_python_package_dir", lambda: str(provider))
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(provider.resolve())])

    assert runtime.activate_configured_openpmd_python_package() == provider
    assert sys.path[0] == str(provider.resolve())
