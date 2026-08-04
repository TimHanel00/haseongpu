ASE solvers
===========

``ASESolver`` is the algorithm role for amplified spontaneous emission.
``MonteCarloASESolver`` is the implementation currently connected to the
native HASEonGPU transport.

.. code-block:: python

   solver = MonteCarloASESolver(
       minRays=100_000,
       maxRays=1_000_000,
       relativeStandardErrorThreshold=0.05,
       adaptiveSteps=4,
       useReflections=True,
       backend="Host_Cpu_CpuSerial",
       openPmdBackend="auto",
   )

Sampling controls
-----------------

``minRays`` is the initial global ray count and ``maxRays`` is the adaptive
upper bound. ``adaptiveSteps`` limits geometric ray-count increases and
``relativeStandardErrorThreshold`` is a dimensionless one-sigma target.
``forwardRayCount`` selects a fixed count and disables adaptive refinement.
The threshold applies to every Tet4 cell; reaching ``maxRays`` does not imply
that every cell met it. See :doc:`uncertainty` for the exact formula, stopping
rule, edge cases, and guidance on choosing a value.

Inspect the cell-wise uncertainty after a completed step:

.. code-block:: python

   state = simulation.step().getLastState()
   print(state.relativeStandardError.max())

Reflection controls are ``useReflections``, ``reflectionMaxIterations``,
``reflectionTolerance``, and ``surfaceReservoirSize``.
``reflectionMaxIterations`` caps reflected surface-reservoir passes,
``reflectionTolerance`` stops them when the remaining reflected source weight
is small enough, and ``surfaceReservoirSize`` sets the number of reservoir
strata retained per face. Forward rays always propagate to a physical
boundary; the old ray-length cutoff and MSE threshold are retired.

Spectral and diagnostic controls
--------------------------------

Native transport consumes the cross-section grid already resolved by the
material frontend. Select its size with
``Material.at(spectralResolution=...)``; the transport request carries the
resulting arrays, not a separate interpolation setting. Increasing numerical
spectral resolution cannot add detail absent from the measured data.
``monochromatic=True`` requests the single-frequency transport mode.

``minSampleRange`` and ``maxSampleRange`` restrict printed cell statistics
to an inclusive index range. They do not crop the mesh or change the all-cell
RSE stopping rule. ``repetitions`` is retained in the openPMD request for
compatibility with older run-control files; the current forward adaptive
estimator accumulates batches in one estimate and does not repeat independent
fixed-count estimates.

Backend controls
----------------

``backend`` selects an Alpaka compute backend returned by
``AlpakaBackends.all()``. ``openPmdBackend`` independently selects the
Python/native transport (``auto``, ``adios``, ``adios-sst``, or ``hdf5`` when
available). See :doc:`../backendSelection` for why these names are not
interchangeable.

``parallelMode="mpi"`` uses ``ranksPerNode`` and the configured MPI launcher.
``numDevices`` caps visible compute devices. ``rngSeed`` makes the Monte Carlo
stream reproducible.

YAML run control
----------------

``MonteCarloASESolver.fromYaml(path, **overrides)`` reads the existing
``experiment`` and ``compute`` sections while keeping mesh, materials, pumps,
and initial state in Python. Python overrides canonically use lower camel case;
the equivalent established snake-case YAML names are also accepted for
compatibility. The YAML wire format deliberately remains snake case:

.. code-block:: yaml

   experiment:
     min_rays: 100000
     max_rays: 1000000
     relative_standard_error_threshold: 0.05
   compute:
     openpmd_backend: auto
     parallel_mode: single

.. code-block:: python

   solver = MonteCarloASESolver.fromYaml(
       "config/hase-phiase.yaml",
       minRays=250_000,
       backend=AlpakaBackends.all()[0],
   )

The default public ray range is ``100_000`` to ``100_000``. Invalid ranges,
negative controls, and unavailable backend configuration are reported before
an ASE request is serialized.

A custom ``ASESolver`` descriptor can be assembled for a future adapter. The
current adapter reports it as unsupported during ``validateBackend()`` rather
than starting transport.
