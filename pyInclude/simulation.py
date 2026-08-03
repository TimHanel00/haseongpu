# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""High-level Python simulation wrapper around pump, ASE, and time stepping."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .laser import (
    MonteCarloPumpSolver,
    PlanarPumpRelay,
    Pump,
    SurfacePumpInjector,
)
from .solvers import ASESolver, PumpSolver
from .openpmd import transport
from hase_units import TIME, Quantity, requireQuantity, units


HASE_CONFIGURE_HINT = "Run `hase-configure` to generate a matching backend/openPMD setup."


def _preferredDefaultBackend():
    try:
        from .alpakaUtils import AlpakaBackends

        backends = AlpakaBackends.all()
    except Exception as exc:
        raise RuntimeError(
            "MonteCarloASESolver.backend is not set and HASEonGPU could not query installed Alpaka "
            f"backends. {HASE_CONFIGURE_HINT}"
        ) from exc
    if not backends:
        raise RuntimeError(
            "MonteCarloASESolver.backend is not set and no Alpaka backend is available. "
            f"{HASE_CONFIGURE_HINT}"
        )
    for marker in ("Host_Cpu_CpuSerial", "CpuSerial"):
        for backend in backends:
            if marker in backend:
                return backend
    return backends[0]


@dataclass
class TimeStepState:
    """Snapshot handed to callbacks after one completed native time step.

    Parameters
    ----------
    step
        One-based completed outer-step index in the current simulation.
    time
        Physical simulation time quantity.
    sampledExcitationFraction
        Dimensionless excitation on backend sample points.
    excitationFraction
        Dimensionless cell-centred excitation, shape ``(numberOfCells,)``.
    phiAse
        ASE photon flux on backend sample points, or ``None`` when unavailable.
    sampledDExcitationDtAse
        ASE contribution to excitation-rate change on sample points, in
        inverse seconds.
    dExcitationDtPump
        Pump contribution to excitation-rate change, in inverse seconds.
    aseResult
        Low-level transport result retained for advanced diagnostics.
    mesh
        :class:`UnstructuredMesh` associated with every cell/point array.
    volumePhiAse
        Optional cell-centred ASE photon flux.
    volumeDExcitationDtAse
        Optional cell-centred ASE excitation-rate contribution.
    volumeStandardError
        Optional absolute Monte Carlo standard error in photon-flux units.
    volumeRelativeStandardError
        Optional dimensionless cell-wise relative standard error.
    volumeTotalRays
        Optional cell-wise deposited ray-visit counts; this is not the global
        launched history count.

    Notes
    -----
    Treat snapshot arrays as read-only analysis data. Mutating them is not a
    supported way to update native simulation state.
    """
    step: int
    """One-based count of completed outer time steps."""
    time: Quantity
    """Physical simulation time at this snapshot."""
    sampledExcitationFraction: np.ndarray
    """Dimensionless upper-state fraction at backend sample points."""
    excitationFraction: np.ndarray
    """Dimensionless cell-centred upper-state fraction."""
    phiAse: np.ndarray | None
    """ASE photon flux at backend sample points, when available."""
    sampledDExcitationDtAse: np.ndarray
    """ASE-induced excitation-rate change at sample points, in ``s^-1``."""
    dExcitationDtPump: np.ndarray
    """Pump-induced cell excitation-rate change, in ``s^-1``."""
    aseResult: object | None
    """Low-level transport result for advanced estimator diagnostics."""
    mesh: object
    """Mesh defining the indexing and coordinates of all spatial arrays."""
    volumePhiAse: np.ndarray | None = None
    """Cell-centred ASE photon flux, when projected by the backend."""
    volumeDExcitationDtAse: np.ndarray | None = None
    """Cell-centred ASE excitation-rate change in ``s^-1``."""
    volumeStandardError: np.ndarray | None = None
    """Absolute one-standard-error estimate for :attr:`volumePhiAse`."""
    volumeRelativeStandardError: np.ndarray | None = None
    """Dimensionless ratio ``volumeStandardError / abs(volumePhiAse)``.

    This is the cell-wise diagnostic compared with
    :attr:`MonteCarloASESolver.relativeStandardErrorThreshold`; it estimates
    sampling noise, not total physical or discretization error.
    """
    volumeTotalRays: np.ndarray | None = None
    """Deposited ray-visit count per cell, not global launched histories."""

    @property
    def dExcitationDtAse(self):
        """Best available ASE excitation-rate array, preferring cell values."""
        return (
            self.volumeDExcitationDtAse
            if self.volumeDExcitationDtAse is not None
            else self.sampledDExcitationDtAse
        )


@dataclass
class MonteCarloASESolver(ASESolver):
    """Numerical and runtime controls for native forward ASE transport.

    Parameters
    ----------
    minRays
        Positive initial global history count for adaptive sampling.
    maxRays
        Global history-count ceiling; must be at least ``minRays``.
    forwardRayCount
        Optional fixed global history count. A non-zero value disables adaptive
        ray-count refinement and overrides the min/max range for transport.
    relativeStandardErrorThreshold
        Dimensionless one-standard-error target applied to every Tet4 cell.
        It controls sampling convergence, not total physical/model error.
    repetitions
        Compatibility run-control value retained on the transport request. The
        current forward estimator accumulates adaptive batches into one result.
    adaptiveSteps
        Maximum geometric increases between ``minRays`` and ``maxRays``.
        Zero requests a single budget.
    useReflections
        Enable reflected-ray surface-reservoir transport.
    reflectionMaxIterations
        Maximum reflected reservoir passes per direct pass.
    reflectionTolerance
        Non-negative stopping fraction for remaining reflected source weight.
    surfaceReservoirSize
        Number of reservoir strata retained per exterior face; reflections
        require a positive value.
    monochromatic
        Enable the native single-frequency transport mode.
    backend
        Alpaka compute backend name. ``None`` selects an available host backend.
    openPmdBackend
        Independent transport backend: ``"auto"``, ``"adios"``,
        ``"adios-sst"``, or ``"hdf5"`` when supported.
    parallelMode
        ``"single"`` or ``"mpi"``.
    numDevices
        Maximum number of accelerator devices used by one run.
    ranksPerNode
        MPI ranks launched per allocated node.
    minSampleRange, maxSampleRange
        Optional inclusive cell-index range used for printed statistics. These
        values do not crop the mesh or change the all-cell RSE stopping rule.
    rngSeed
        Optional reproducible unsigned seed for ASE history generation.

    Examples
    --------
    ``MonteCarloASESolver(minRays=100_000, maxRays=1_600_000,``
    ``relativeStandardErrorThreshold=0.05)``
    """

    minRays: int = 100_000
    """Initial global Monte Carlo history budget for adaptive sampling."""
    maxRays: int = 100_000
    """Maximum global history budget allowed during adaptive sampling."""
    forwardRayCount: int | None = None
    """Fixed global history count; a non-zero value disables adaptation."""
    relativeStandardErrorThreshold: float = 0.1
    """Dimensionless target ``SE(phiAse) / abs(phiAse)`` applied cell by cell.

    Adaptive sampling grows the global history budget until every eligible
    Tet4 cell meets this target or :attr:`maxRays` is reached. This threshold
    controls Monte Carlo sampling uncertainty only; it does not bound mesh,
    spectral, material-data, or model error.
    """
    repetitions: int = 4
    """Run-control repetition value carried in the transport request."""
    adaptiveSteps: int = 4
    """Maximum number of geometric budget increases from min to max rays."""
    useReflections: bool = False
    """Whether rays deposited in the surface reservoir are propagated."""
    reflectionMaxIterations: int = 40
    """Maximum reflected-reservoir passes per direct transport pass."""
    reflectionTolerance: float = 1.0e-4
    """Stopping fraction for reflected source weight remaining in reservoirs."""
    surfaceReservoirSize: int = 32
    """Number of reflected-ray reservoir strata stored per exterior face."""
    monochromatic: bool = False
    """Whether native transport uses its single-frequency mode."""
    backend: str | None = None
    """Alpaka compute-backend name; ``None`` chooses an available host backend."""
    openPmdBackend: str = "auto"
    """openPMD transport implementation, independent of the compute backend."""
    parallelMode: str = "single"
    """Process topology: ``"single"`` or ``"mpi"``."""
    numDevices: int = 1
    """Maximum accelerator devices used by one simulation run."""
    ranksPerNode: int = 1
    """MPI processes launched on each allocated node."""
    minSampleRange: int | None = None
    """First cell id included in printed diagnostic statistics."""
    maxSampleRange: int | None = None
    """Last cell id included in printed diagnostic statistics."""
    rngSeed: int | None = None
    """Unsigned random seed for reproducible ASE history generation."""

    def __post_init__(self):
        if int(self.minRays) <= 0:
            raise ValueError("minRays must be positive")
        if int(self.maxRays) < int(self.minRays):
            raise ValueError("maxRays must be greater than or equal to minRays")
        if self.forwardRayCount is not None and int(self.forwardRayCount) < 0:
            raise ValueError("forwardRayCount must be non-negative")
        if int(self.adaptiveSteps) < 0:
            raise ValueError("adaptiveSteps must be non-negative")
        if self.parallelMode not in {"single", "mpi"}:
            raise ValueError("parallelMode must be 'single' or 'mpi'")

    @classmethod
    def fromYaml(cls, filename, **overrides):
        """Construct solver controls from ``experiment`` and ``compute`` YAML.

        Parameters
        ----------
        filename
            YAML path containing zero or more public dataclass field names in
            ``experiment`` and ``compute`` mappings.
        **overrides
            Lower-camel-case Python constructor values applied after the file,
            for example ``minRays=200_000``. YAML itself retains its established
            snake-case wire keys.

        Returns
        -------
        MonteCarloASESolver
            Validated descriptor; no backend process is launched.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("MonteCarloASESolver YAML configuration requires PyYAML") from exc
        with Path(filename).open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        if not isinstance(document, dict):
            raise ValueError("ASE configuration must contain a mapping")
        values = {}
        yamlNames = {
            "min_rays": "minRays",
            "max_rays": "maxRays",
            "forward_ray_count": "forwardRayCount",
            "relative_standard_error_threshold": "relativeStandardErrorThreshold",
            "repetitions": "repetitions",
            "adaptive_steps": "adaptiveSteps",
            "use_reflections": "useReflections",
            "reflection_max_iterations": "reflectionMaxIterations",
            "reflection_tolerance": "reflectionTolerance",
            "surface_reservoir_size": "surfaceReservoirSize",
            "monochromatic": "monochromatic",
            "backend": "backend",
            "openpmd_backend": "openPmdBackend",
            "parallel_mode": "parallelMode",
            "num_devices": "numDevices",
            "ranks_per_node": "ranksPerNode",
            "min_sample_range": "minSampleRange",
            "max_sample_range": "maxSampleRange",
            "rng_seed": "rngSeed",
        }
        allowed = set(cls.__dataclass_fields__)
        for section_name in ("experiment", "compute"):
            section = document.get(section_name, {})
            if not isinstance(section, dict):
                raise ValueError(f"ASE configuration section '{section_name}' must be a mapping")
            unknown = sorted(set(section) - set(yamlNames))
            if unknown:
                raise ValueError(f"unknown ASE configuration key '{unknown[0]}'")
            values.update((yamlNames[name], value) for name, value in section.items())
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise TypeError(f"unknown MonteCarloASESolver override '{unknown[0]}'")
        values.update(overrides)
        return cls(**values)

    def transportAttributes(self, numberOfSamples):
        """Serialize public controls into the HASE openPMD attribute schema.

        Parameters
        ----------
        numberOfSamples
            Number of cells/samples used to resolve an omitted
            ``maxSampleRange``.

        Returns
        -------
        dict[str, object]
            Camel-case internal values consumed by the transport encoder.

        Notes
        -----
        This is mainly an adapter-facing method; normal user code passes the
        solver directly to :class:`Simulation`.
        """
        attributes = {
            "minRays": int(self.minRays),
            "maxRays": int(self.maxRays),
            "propagationMode": "forward",
            "forwardRayCount": 0 if self.forwardRayCount is None else int(self.forwardRayCount),
            "relativeStandardErrorThreshold": float(self.relativeStandardErrorThreshold),
            "reflectionMaxIterations": int(self.reflectionMaxIterations),
            "reflectionTolerance": float(self.reflectionTolerance),
            "surfaceReservoirSize": int(self.surfaceReservoirSize),
            "repetitions": int(self.repetitions),
            "adaptiveSteps": int(self.adaptiveSteps),
            "useReflections": bool(self.useReflections),
            "monochromatic": bool(self.monochromatic),
            "backend": _preferredDefaultBackend() if self.backend is None else self.backend,
            "maxGpus": int(self.numDevices),
            "parallelMode": self.parallelMode,
            "minSampleRange": 0 if self.minSampleRange is None else int(self.minSampleRange),
            "maxSampleRange": numberOfSamples - 1 if self.maxSampleRange is None else int(self.maxSampleRange),
        }
        if self.rngSeed is not None:
            attributes["rngSeed"] = int(self.rngSeed)
        return attributes

    def launchOptions(self):
        """Return process-launch options implied by ``parallelMode``.

        Returns an empty mapping for single-process execution. MPI mode returns
        an ``mpiexec`` command prefix and shared workspace directory.
        """
        if self.parallelMode != "mpi":
            return {}
        if int(self.ranksPerNode) < 1:
            raise ValueError("ranksPerNode must be positive for MPI execution")
        return {
            "command_prefix": [
                "mpiexec",
                *shlex.split(os.environ.get("HASE_MPIEXEC_EXTRA_ARGS", "")),
                "-npernode",
                str(int(self.ranksPerNode)),
            ],
            "workspace_dir": Path.cwd() / "IO" / "phiase_mpi",
        }


class Simulation:
    """Compose a Tet4 mesh, physical definitions, layouts, and numerical solvers.

    Parameters
    ----------
    mesh
        Tet4 :class:`UnstructuredMesh` owning every later selection.
    aseSolver
        Descriptor implementing the :class:`ASESolver` role.
    pumpSolver
        Descriptor implementing the :class:`PumpSolver` role.
    timeIntegrator
        Time-integrator descriptor or valid native solver-name string.
    timeStepSize
        Positive physical time quantity for one outer step.
    initialState
        Dimensionless cell excitation specification.
    maxSteps
        Optional positive outer-step limit.
    maxTime
        Optional positive physical time limit.
    enableAse
        Whether native steps evaluate ASE transport.
    prePump
        Whether pumping is applied before the first time integration update.
    reportTimings
        Request native timing diagnostics.

    Notes
    -----
    :meth:`resolveProblem` assembles and validates multiple materials and internal
    interfaces as backend-neutral, inspectable Python tables; it does not
    compile Python or C++ code and launches no native process. The current
    native adapter accepts only one isotropic active material and no internal
    material interfaces. Registrations freeze when execution initializes.
    """

    mesh: object
    """Tet4 mesh that owns all spatial selections and array indexing."""
    aseSolver: ASESolver
    """Numerical descriptor controlling amplified-spontaneous-emission transport."""
    pumpSolver: PumpSolver
    """Numerical descriptor controlling pump-ray transport."""
    timeIntegrator: object
    """Time-integration descriptor or native integrator-name string."""
    timeStepSize: Quantity
    """Positive physical duration advanced by one outer simulation step."""
    initialState: object
    """Initial dimensionless upper-state population specification."""
    maxSteps: int | None
    """Optional upper limit on completed outer steps."""
    maxTime: Quantity | None
    """Optional physical end time used by :meth:`runUntil`."""
    enableAse: bool
    """Whether each native time step evaluates ASE transport."""
    prePump: bool
    """Whether pump transport is applied before the first state update."""
    reportTimings: bool
    """Whether the native run emits detailed timing diagnostics."""

    def __init__(
        self,
        *,
        mesh,
        aseSolver,
        pumpSolver,
        timeIntegrator,
        timeStepSize,
        initialState,
        maxSteps=None,
        maxTime=None,
        enableAse=True,
        prePump=False,
        reportTimings=False,
    ):
        from .mesh import UnstructuredMesh
        from .problem import InitialState

        if not isinstance(mesh, UnstructuredMesh):
            raise TypeError("mesh must be UnstructuredMesh")
        if not isinstance(aseSolver, ASESolver):
            raise TypeError("aseSolver must implement the ASESolver role")
        if not isinstance(pumpSolver, PumpSolver):
            raise TypeError("pumpSolver must implement the PumpSolver role")
        if not isinstance(initialState, InitialState):
            raise TypeError("initialState must be InitialState")
        if not isinstance(timeIntegrator, str) and not hasattr(timeIntegrator, "name"):
            raise TypeError("timeIntegrator must be a solver name or descriptor with a name")
        timeStepSize = requireQuantity(timeStepSize, TIME, "timeStepSize")
        if not np.isfinite(timeStepSize.siValue) or float(timeStepSize.siValue) <= 0.0:
            raise ValueError("timeStepSize must be finite and positive")
        if maxSteps is not None and int(maxSteps) <= 0:
            raise ValueError("maxSteps must be positive")
        maxTime = requireQuantity(maxTime, TIME, "maxTime", allowNone=True)
        if maxTime is not None and (not np.isfinite(maxTime.siValue) or float(maxTime.siValue) <= 0.0):
            raise ValueError("maxTime must be finite and positive")
        self.mesh = mesh
        self.aseSolver = aseSolver
        self.pumpSolver = pumpSolver
        self.timeIntegrator = timeIntegrator
        self.timeStepSize = timeStepSize
        self.initialState = initialState
        self.maxSteps = None if maxSteps is None else int(maxSteps)
        self.maxTime = maxTime
        self.enableAse = bool(enableAse)
        self.prePump = bool(prePump)
        self.reportTimings = bool(reportTimings)
        self._material_registrations = []
        self._boundary_registrations = []
        self._interface_registrations = []
        self._pump_registrations = []
        self._init_callbacks = []
        self._step_callbacks = []
        self._initialized = False
        self._compiled_problem = None
        self._current_step = 0
        self._current_time = Quantity(0.0, units.s)
        self._last_state = None

    def _require_mutable(self, what):
        if self._initialized:
            raise RuntimeError(f"{what} must be configured before the simulation is initialized")

    def addMaterial(self, material, *, domains):
        """Assign one resolved material condition to selected volume cells.

        Parameters
        ----------
        material
            :class:`MaterialCondition` selected for this run.
        domains
            Volume-kind :class:`MeshSelection` from ``simulation.mesh``.

        Returns
        -------
        Simulation
            ``self`` for fluent configuration.
        """
        self._require_mutable("materials")
        self._material_registrations.append((material, domains))
        self._compiled_problem = None
        return self

    def addBoundary(self, boundary, *, domains):
        """Assign an exterior optical model to selected boundary faces.

        Parameters
        ----------
        boundary
            Descriptor implementing :class:`ExteriorBoundaryModel`.
        domains
            Surface-kind selection from this simulation's mesh. Every exterior
            face must be covered exactly once before problem resolution.

        Returns
        -------
        Simulation
            ``self`` for fluent configuration.
        """
        self._require_mutable("boundaries")
        self._boundary_registrations.append((boundary, domains))
        self._compiled_problem = None
        return self

    def addInterface(self, interface, *, between):
        """Assign a model between two resolved material conditions.

        Parameters
        ----------
        interface
            :class:`MaterialInterfaceModel` descriptor.
        between
            Two :class:`MaterialCondition` objects already used by material
            registrations. Order is not physically directional.

        Returns
        -------
        Simulation
            ``self`` for fluent configuration.
        """
        self._require_mutable("material interfaces")
        self._interface_registrations.append((interface, tuple(between)))
        self._compiled_problem = None
        return self

    def addPump(self, pump, injectionMethod, *, relays=()):
        """Register physical pump light, its aperture, and optional relays.

        Parameters
        ----------
        pump
            Physical light definition: total incident power, wavelength-power
            distribution, transverse profile, and launch-angle distribution.
            It says what light is injected, but not where it enters the mesh.
        injectionMethod
            Geometric coupling between ``pump`` and this simulation's mesh.
            A :class:`SurfacePumpInjector` selects exterior triangular faces as
            the launch aperture. Their oriented normals establish the local
            inward direction; their areas and local coordinates provide the
            surface over which the transverse profile is normalized. It does
            not define absorption or reflection after a ray reaches a boundary;
            configure that independently with :meth:`addBoundary`.
        relays
            Optional re-imaging routes for rays leaving one selected plane and
            re-entering another. Relays are not additional pump sources and
            their transmission scales only the routed ray weight.

        Returns
        -------
        Simulation
            ``self`` for fluent configuration.
        """
        self._require_mutable("pumps")
        if not isinstance(pump, Pump):
            raise TypeError("pump must be Pump")
        if not isinstance(injectionMethod, SurfacePumpInjector):
            raise TypeError("injectionMethod must be SurfacePumpInjector")
        relays = tuple(relays)
        if not all(isinstance(relay, PlanarPumpRelay) for relay in relays):
            raise TypeError("relays must contain PlanarPumpRelay values")
        if injectionMethod.surface.mesh is not self.mesh:
            raise ValueError("pump injection surface belongs to a different mesh")
        if any(relay.exitSurface.mesh is not self.mesh for relay in relays):
            raise ValueError("pump relay surface belongs to a different mesh")
        self._pump_registrations.append((pump, injectionMethod, relays))
        return self

    def resolveProblem(self):
        """Validate physical coverage and return a backend-neutral problem.

        Resolution creates dense material/boundary/interface tables and the
        initial cell state. It performs no native launch and does not require
        that the current adapter support every valid frontend feature.
        """
        from .problem import resolve_problem

        self._compiled_problem = resolve_problem(
            self.mesh,
            self._material_registrations,
            self._boundary_registrations,
            self._interface_registrations,
            self.initialState,
        )
        return self._compiled_problem

    def validateBackend(self):
        """Resolve the problem and reject features unsupported by the native adapter."""
        problem = self.resolveProblem().requireBackendSupport()
        self._require_solver_support()
        return problem

    def _require_solver_support(self):
        unsupported = []
        if not isinstance(self.aseSolver, MonteCarloASESolver):
            unsupported.append(f"ASE solver {type(self.aseSolver).__name__}")
        if not isinstance(self.pumpSolver, MonteCarloPumpSolver):
            unsupported.append(f"pump solver {type(self.pumpSolver).__name__}")
        if unsupported:
            raise NotImplementedError(
                "the current HASEonGPU backend adapter supports only Monte Carlo solvers; unsupported "
                + "; ".join(unsupported)
            )

    def _initialize(self):
        if self._initialized:
            return
        for callback, args, kwargs in self._init_callbacks:
            callback(self, *args, **kwargs)
        self._compiled_problem = self.resolveProblem()
        self._compiled_problem.requireBackendSupport()
        self._require_solver_support()
        self._initialized = True

    def onInit(self, callback, *args, **kwargs):
        """Register a callback executed once immediately before problem resolution.

        Parameters
        ----------
        callback
            Callable receiving the live :class:`Simulation`.
        *args, **kwargs
            Additional values forwarded after the simulation argument.

        It may complete configuration, but all registrations freeze after
        initialization finishes. Returns ``self``.
        """
        self._require_mutable("initialization callbacks")
        self._init_callbacks.append((callback, args, kwargs))
        return self

    def onStep(self, callback, *args, **kwargs):
        """Register a callback for every completed native snapshot.

        Parameters
        ----------
        callback
            Callable receiving a completed :class:`TimeStepState`.
        *args, **kwargs
            Additional values forwarded after the state argument.

        Callback speed may delay file-based runs; streaming reception itself
        is drained independently. Returns ``self``.
        """
        self._step_callbacks.append((callback, args, kwargs))
        return self

    def step(self, numberOfSteps=1, *, pumpSteps=None):
        """Advance a positive number of outer time steps.

        Parameters
        ----------
        numberOfSteps
            Positive number of outer steps to execute.
        pumpSteps
            Optional non-negative number of these steps that include pumping.
            ``None`` uses the pump solver/run-control default.

        Returns
        -------
        Simulation
            ``self`` after callbacks and state counters are updated.
        """
        if int(numberOfSteps) <= 0:
            raise ValueError("numberOfSteps must be positive")
        if pumpSteps is not None and int(pumpSteps) < 0:
            raise ValueError("pumpSteps must be non-negative")
        self.validateBackend()
        if not self._pump_registrations:
            raise ValueError(
                "the current backend requires at least one pump registered with addPump"
            )
        self._initialize()
        previous_step = self._current_step
        previous_time = float(self._current_time.toValue(units.s))
        states = transport.run_simulation(
            self,
            steps=int(numberOfSteps),
            pump_steps=pumpSteps,
            transport=self.aseSolver.openPmdBackend,
            **self.aseSolver.launchOptions(),
        )
        for raw_state in states:
            state = TimeStepState(
                step=previous_step + int(raw_state.step),
                time=Quantity(previous_time + float(raw_state.time), units.s),
                sampledExcitationFraction=np.asarray(raw_state.betaCells, dtype=np.float64).copy(),
                excitationFraction=np.asarray(raw_state.betaVolume, dtype=np.float64).copy(),
                phiAse=np.asarray(raw_state.phiAse, dtype=np.float64).copy(),
                sampledDExcitationDtAse=np.asarray(raw_state.dndtAse, dtype=np.float64).copy(),
                dExcitationDtPump=np.asarray(raw_state.dndtPump, dtype=np.float64).copy(),
                aseResult=raw_state.aseResult,
                mesh=self.mesh,
                volumePhiAse=None if getattr(raw_state, "volumePhiAse", None) is None else np.asarray(raw_state.volumePhiAse).copy(),
                volumeDExcitationDtAse=None if getattr(raw_state, "volumeDndtAse", None) is None else np.asarray(raw_state.volumeDndtAse).copy(),
                volumeStandardError=None if getattr(raw_state, "volumeStandardError", None) is None else np.asarray(raw_state.volumeStandardError).copy(),
                volumeRelativeStandardError=None if getattr(raw_state, "volumeRelativeStandardError", None) is None else np.asarray(raw_state.volumeRelativeStandardError).copy(),
                volumeTotalRays=None if getattr(raw_state, "volumeTotalRays", None) is None else np.asarray(raw_state.volumeTotalRays).copy(),
            )
            self._last_state = state
            self._current_step = state.step
            self._current_time = state.time
            for callback, args, kwargs in self._step_callbacks:
                callback(state, *args, **kwargs)
        return self

    def runUntil(self, maxTime=None):
        """Advance whole steps until the requested physical time is reached.

        Parameters
        ----------
        maxTime
            Positive time quantity. ``None`` uses the constructor's
            ``maxTime``.

        Returns
        -------
        Simulation
            ``self``; no step is run when the target is already reached.
        """
        target = self.maxTime if maxTime is None else maxTime
        target = requireQuantity(target, TIME, "maxTime", allowNone=True)
        if target is None or not np.isfinite(target.siValue) or float(target.siValue) <= 0.0:
            raise ValueError("runUntil requires a finite, positive maxTime")
        remaining = float(target.toValue(units.s)) - float(self._current_time.toValue(units.s))
        dt = float(self.timeStepSize.toValue(units.s))
        steps = max(0, int(np.ceil((remaining - 0.5 * dt) / dt)))
        if steps:
            self.step(steps)
        return self

    def getLastState(self):
        """Return the most recent :class:`TimeStepState`.

        Raises ``RuntimeError`` before the first completed native step.
        """
        if self._last_state is None:
            raise RuntimeError("simulation has not completed a time step yet")
        return self._last_state

    @property
    def currentStep(self):
        """Number of completed outer time steps."""
        return self._current_step

    @property
    def currentTime(self):
        """Physical time :class:`Quantity` after completed steps."""
        return self._current_time

    @property
    def resolvedProblem(self):
        """Cached :class:`ResolvedProblem`, resolving on first access."""
        if self._compiled_problem is None:
            self._compiled_problem = self.resolveProblem()
        return self._compiled_problem

    @property
    def pumpRegistrations(self):
        """Immutable tuple of ``(pump, injector, relays)`` registrations."""
        return tuple(self._pump_registrations)

    @property
    def pumps(self):
        """Physical :class:`Pump` objects in registration order."""
        return tuple(pump for pump, _injector, _relays in self._pump_registrations)
