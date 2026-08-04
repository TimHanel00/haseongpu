# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""High-level Python simulation wrapper around pump, ASE, and time stepping."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
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

SIMULATION_OUTPUT_FIELDS = (
    "beta_volume",
    "phi_ase",
    "standard_error",
    "relative_standard_error",
    "total_rays",
    "dndt_ase",
    "dndt_pump",
)
SIMULATION_CONTROL_FIELDS = ("beta_volume",)


def autonomousFinal(numberOfSteps):
    """Return the final completed-step index as an autonomous output schedule."""
    numberOfSteps = int(numberOfSteps)
    if numberOfSteps <= 0:
        raise ValueError("numberOfSteps must be positive")
    return (numberOfSteps,)


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
    excitationFraction
        Dimensionless cell-centred excitation, shape ``(numberOfCells,)``.
    phiAse
        Cell-centred ASE photon flux, or ``None`` when not selected.
    standardError
        Cell-centred Monte Carlo absolute standard error.
    relativeStandardError
        Cell-centred relative standard error.
    totalRays
        Cell-centred deposited ray-visit counts.
    dExcitationDtAse
        Cell-centred ASE contribution to excitation-rate change.
    dExcitationDtPump
        Cell-centred pump contribution to excitation-rate change.
    aseResult
        Low-level transport result retained for advanced diagnostics.
    mesh
        :class:`UnstructuredMesh` associated with every cell/point array.
    Notes
    -----
    Treat snapshot arrays as read-only analysis data. Mutating them is not a
    supported way to update native simulation state.
    """
    step: int
    """One-based count of completed outer time steps."""
    time: Quantity
    """Physical simulation time at this snapshot."""
    excitationFraction: np.ndarray | None
    """Dimensionless cell-centred upper-state fraction."""
    phiAse: np.ndarray | None
    """Cell-centred ASE photon flux, when selected."""
    standardError: np.ndarray | None
    """Cell-centred absolute Monte Carlo standard error."""
    relativeStandardError: np.ndarray | None
    """Cell-centred relative Monte Carlo standard error."""
    totalRays: np.ndarray | None
    """Cell-centred deposited ray-visit count."""
    dExcitationDtAse: np.ndarray | None
    """Cell-centred ASE excitation-rate change in ``s^-1``."""
    dExcitationDtPump: np.ndarray | None
    """Pump-induced cell excitation-rate change, in ``s^-1``."""
    aseResult: object | None
    """Low-level transport result for advanced estimator diagnostics."""
    mesh: object
    """Mesh defining the indexing and coordinates of all spatial arrays."""


def _readOnlyArray(value, dtype=None):
    """Return a NumPy view suitable for a read-only snapshot field."""
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


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
            Python constructor values applied after the file. Lower-camel-case
            names such as ``minRays=200_000`` are canonical; the corresponding
            established snake-case YAML names remain accepted for compatibility.

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
        normalizedOverrides = {}
        for name, value in overrides.items():
            publicName = yamlNames.get(name, name)
            if publicName in normalizedOverrides:
                raise TypeError(f"duplicate MonteCarloASESolver override '{publicName}'")
            normalizedOverrides[publicName] = value
        unknown = sorted(set(normalizedOverrides) - allowed)
        if unknown:
            raise TypeError(f"unknown MonteCarloASESolver override '{unknown[0]}'")
        values.update(normalizedOverrides)
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
    executionMode
        ``"autonomous"`` for a backend-owned run or ``"synchronized-debug"``
        for a snapshot/control exchange after every completed step.
    outputSteps
        Optional strictly increasing completed-step indices emitted by an
        autonomous run. Omit to emit every completed step.
    outputFields
        Cell-centred fields selected once before backend initialization.
    controlFields
        Dynamic openPMD fields accepted in synchronized-debug mode. The first
        implementation supports only ``"beta_volume"``.

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
    executionMode: str
    """Backend-owned autonomous execution or synchronized debug exchange."""
    outputSteps: tuple[int, ...] | None
    """Selected one-based completed-step indices, or every step when omitted."""
    outputFields: tuple[str, ...]
    """Cell-centred fields copied into each selected output snapshot."""
    controlFields: tuple[str, ...]
    """Dynamic fields accepted from synchronized-debug callbacks."""

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
        executionMode="autonomous",
        outputSteps=None,
        outputFields=None,
        controlFields=None,
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
        if executionMode not in {"autonomous", "synchronized-debug"}:
            raise ValueError("executionMode must be 'autonomous' or 'synchronized-debug'")
        outputSteps = None if outputSteps is None else tuple(int(step) for step in outputSteps)
        outputFields = tuple(
            SIMULATION_OUTPUT_FIELDS if outputFields is None else (str(field) for field in outputFields)
        )
        controlFields = tuple(() if controlFields is None else (str(field) for field in controlFields))
        if outputSteps is not None and not outputSteps:
            raise ValueError("outputSteps must be omitted or contain at least one completed-step index")
        if outputSteps is not None and any(step <= 0 for step in outputSteps):
            raise ValueError("outputSteps must contain positive completed-step indices")
        if outputSteps is not None and tuple(sorted(set(outputSteps))) != outputSteps:
            raise ValueError("outputSteps must be strictly increasing and unique")
        if not outputFields:
            raise ValueError("outputFields must contain at least one field")
        unknownOutputs = sorted(set(outputFields) - set(SIMULATION_OUTPUT_FIELDS))
        if unknownOutputs:
            raise ValueError(f"unsupported outputFields: {unknownOutputs}")
        if len(set(outputFields)) != len(outputFields):
            raise ValueError("outputFields must be unique")
        unknownControls = sorted(set(controlFields) - set(SIMULATION_CONTROL_FIELDS))
        if unknownControls:
            raise ValueError(f"unsupported controlFields: {unknownControls}")
        if len(set(controlFields)) != len(controlFields):
            raise ValueError("controlFields must be unique")
        if executionMode == "autonomous" and controlFields:
            raise ValueError("controlFields require executionMode='synchronized-debug'")
        if executionMode == "synchronized-debug" and outputSteps is not None:
            raise ValueError("synchronized-debug emits every completed step; outputSteps must be omitted")
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
        self.executionMode = executionMode
        self.outputSteps = outputSteps
        self.outputFields = outputFields
        self.controlFields = controlFields
        self._material_registrations = []
        self._boundary_registrations = []
        self._interface_registrations = []
        self._pump_registrations = []
        self._init_callbacks = []
        self._step_callbacks = []
        self._control_callbacks = []
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

    def onControl(self, callback, *args, **kwargs):
        """Register a synchronized-debug control callback.

        The callback receives the completed :class:`TimeStepState` and returns
        a mapping from a configured ``controlFields`` name to its next
        cell-centred values. Returning ``None`` sends no fields for that
        exchange. Control callbacks run only between non-final debug steps.
        """
        self._require_mutable("control callbacks")
        self._control_callbacks.append((callback, args, kwargs))
        return self

    def _materializeState(self, rawState, previousStep, previousTime):
        state = TimeStepState(
            step=previousStep + int(rawState.step),
            time=Quantity(previousTime + float(rawState.time), units.s),
            excitationFraction=_readOnlyArray(rawState.betaVolume, np.float64),
            phiAse=_readOnlyArray(rawState.phiAse, np.float64),
            standardError=_readOnlyArray(rawState.standardError, np.float64),
            relativeStandardError=_readOnlyArray(rawState.relativeStandardError, np.float64),
            totalRays=_readOnlyArray(rawState.totalRays),
            dExcitationDtAse=_readOnlyArray(rawState.dndtAse, np.float64),
            dExcitationDtPump=_readOnlyArray(rawState.dndtPump, np.float64),
            aseResult=rawState.aseResult,
            mesh=self.mesh,
        )
        self._last_state = state
        self._current_step = state.step
        self._current_time = state.time
        for callback, args, kwargs in self._step_callbacks:
            callback(state, *args, **kwargs)

        controls = {}
        for callback, args, kwargs in self._control_callbacks:
            update = callback(state, *args, **kwargs)
            if update is None:
                continue
            if not isinstance(update, Mapping):
                raise TypeError("onControl callbacks must return a field mapping or None")
            duplicate = set(controls).intersection(update)
            if duplicate:
                raise ValueError(f"multiple onControl callbacks returned {sorted(duplicate)}")
            controls.update(update)
        unknown = sorted(set(controls) - set(self.controlFields))
        if unknown:
            raise ValueError(f"onControl returned fields not configured in controlFields: {unknown}")
        return controls

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
        if self._current_step != 0:
            raise RuntimeError(
                "a compiled Simulation is initialized and run once; pass the complete autonomous step count "
                "to the first step() call"
            )
        if self.outputSteps is not None and self.outputSteps[-1] > int(numberOfSteps):
            raise ValueError("outputSteps cannot exceed numberOfSteps")
        if self.executionMode == "autonomous" and self._control_callbacks:
            raise ValueError("onControl callbacks require executionMode='synchronized-debug'")
        self.validateBackend()
        if not self._pump_registrations:
            raise ValueError(
                "the current backend requires at least one pump registered with addPump"
            )
        self._initialize()
        previous_step = self._current_step
        previous_time = float(self._current_time.toValue(units.s))
        consumedStates = []

        def consume(rawState):
            controls = self._materializeState(rawState, previous_step, previous_time)
            consumedStates.append(rawState)
            return controls

        states = transport.run_simulation(
            self,
            steps=int(numberOfSteps),
            pump_steps=pumpSteps,
            transport=self.aseSolver.openPmdBackend,
            on_state=consume,
            **self.aseSolver.launchOptions(),
        )
        if not consumedStates:
            for rawState in states:
                consume(rawState)
        self._current_step = previous_step + int(numberOfSteps)
        self._current_time = Quantity(
            previous_time + int(numberOfSteps) * float(self.timeStepSize.toValue(units.s)),
            units.s,
        )
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
