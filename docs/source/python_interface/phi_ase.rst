ASE solvers
===========

``ASESolver`` is the algorithm role for amplified spontaneous emission.
``MonteCarloASESolver`` is the implementation currently connected to the
native HASEonGPU transport.

.. code-block:: python

   solver = MonteCarloASESolver(
       min_rays=100_000,
       max_rays=1_000_000,
       relative_standard_error_threshold=0.05,
       repetitions=2,
       adaptive_steps=4,
       use_reflections=True,
       backend="Host_Cpu_CpuSerial",
       openpmd_backend="auto",
   )

Sampling controls
-----------------

``min_rays`` is the initial global ray count and ``max_rays`` is the adaptive
upper bound. ``adaptive_steps`` limits geometric ray-count increases and
``repetitions`` limits repeated estimates at one count.
``relative_standard_error_threshold`` is a dimensionless one-sigma target.
``forward_ray_count`` selects a fixed count and disables adaptive refinement.

Reflection controls are ``use_reflections``, ``reflection_max_iterations``,
``reflection_tolerance``, and ``surface_reservoir_size``. Forward rays always
propagate to a physical boundary; the old ray-length cutoff and MSE threshold
are retired.

Backend controls
----------------

``backend`` selects an Alpaka compute backend returned by
``AlpakaBackends.all()``. ``openpmd_backend`` independently selects the
Python/native transport (``auto``, ``adios``, ``adios-sst``, or ``hdf5`` when
available). See :doc:`../backendSelection` for why these names are not
interchangeable.

``parallel_mode="mpi"`` uses ``ranks_per_node`` and the configured MPI launcher.
``num_devices`` caps visible compute devices. ``rng_seed`` makes the Monte Carlo
stream reproducible.

YAML run control
----------------

``MonteCarloASESolver.from_yaml(path, **overrides)`` reads the existing
``experiment`` and ``compute`` sections while keeping mesh, materials, pumps,
and initial state in Python. Public constructor override names are snake case:

.. code-block:: python

   solver = MonteCarloASESolver.from_yaml(
       "config/hase-phiase.yaml",
       min_rays=250_000,
       backend=AlpakaBackends.all()[0],
   )

The default public ray range is ``100_000`` to ``100_000``. Invalid ranges,
negative controls, and unavailable backend configuration are reported before
an ASE request is serialized.

A custom ``ASESolver`` descriptor can be assembled for a future adapter. The
current adapter reports it as unsupported during ``validate_backend()`` rather
than starting transport.
