Utilities
=========

Time integration
----------------

The public time integrators are lightweight descriptors for native compiled
algorithms:

.. code-block:: python

   from HASEonGPU import FrozenPhiAseRungeKutta4

   simulation = Simulation(
       # ...
       time_integrator=FrozenPhiAseRungeKutta4(),
   )

Available descriptors are:

* ``ExplicitEuler()``;
* ``Heun()``;
* ``Midpoint()``;
* ``RungeKutta4()``;
* ``FrozenPhiAseRungeKutta4()``, which reuses one ASE evaluation across RK4
  stages;
* ``ImplicitEuler(iterations=8, tolerance=1e-10)``;
* ``ExponentialEuler()``.

Python serializes the descriptor's ``name`` into native run control. A valid
solver-name string is also accepted. Custom Python integration callables are
not executed by compiled simulations.

Backend discovery
-----------------

``AlpakaBackends.all()`` lists compute backends available in the installed
runtime:

.. code-block:: python

   from HASEonGPU import AlpakaBackends, MonteCarloASESolver

   available = AlpakaBackends.all()
   if not available:
       raise RuntimeError("HASEonGPU was built without an available backend")
   ase_solver = MonteCarloASESolver(backend=available[0])

``known()`` is an alias for ``all()``. Names that are valid Python identifiers
are also class attributes. These are Alpaka compute names, not openPMD backend
names; see :doc:`../backendSelection`.

State export
------------

``writeParaviewState(state, output_dir, ...)`` appends a completed
``TimeStepState`` to an openPMD series and writes a small ``.pmd`` handle for
ParaView. It can be registered directly as a callback:

.. code-block:: python

   simulation.on_step(writeParaviewState, "output/openpmd")

For custom analysis, consume the NumPy views directly:

.. code-block:: python

   def save_excitation(state, output_dir):
       np.save(output_dir / f"beta-{state.step:06d}.npy", state.excitation_fraction)

   simulation.on_step(save_excitation, output_dir)

The former ``vtkWedge`` and ``calcGainFromState`` helpers are not part of the
public composition API. Repository-owned compatibility regressions still use
them privately for historical wedge references.

Low-level openPMD schema helpers
--------------------------------

The public namespace still exposes schema-building objects such as
``PrimitiveFieldSpec``, ``PointSchema``, ``TriangleSchema``, and
``PrismSchema`` for advanced openPMD tooling. They do not replace
``UnstructuredMesh`` or the material/layout API and are not required for a
normal simulation.
