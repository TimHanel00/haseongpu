ASE solvers
===========

``ASESolver`` is the extension role for algorithms that evaluate amplified
spontaneous emission. ``MonteCarloASESolver`` is the implementation currently
wired to HASEonGPU's native backend.

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

``MonteCarloASESolver.from_yaml(path, **overrides)`` reads existing run-control
YAML. Public constructor names are snake_case. The Alpaka compute backend and
openPMD transport backend remain independent choices.

A custom ``ASESolver`` descriptor can be composed with ``Simulation``. The
current backend adapter reports it as unsupported before launch until a native
adapter for that algorithm is implemented.
