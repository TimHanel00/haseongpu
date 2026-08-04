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
       timeIntegrator=FrozenPhiAseRungeKutta4(),
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
   aseSolver = MonteCarloASESolver(backend=available[0])

``known()`` is an alias for ``all()``. Names that are valid Python identifiers
are also class attributes. These are Alpaka compute names, not openPMD backend
names; see :doc:`../backendSelection`.

State export
------------

``writeParaviewState(state, output_dir, ...)`` appends a completed
``TimeStepState`` to an openPMD series and writes a small ``.pmd`` handle for
ParaView. It can be registered directly as a callback:

.. code-block:: python

   simulation.onStep(writeParaviewState, "output/openpmd")

When ``pattern`` is omitted, the exporter inspects the active Python
``openpmd_api`` provider. It uses an ADIOS2 ``.bp`` series when ADIOS2 is
available and otherwise falls back to HDF5 ``.h5``. An explicitly supplied
pattern remains authoritative, so its suffix must name a backend supported by
that provider.

``writeVtkState(file_name, state, ...)`` writes the same public state as a
legacy ASCII Tet4 VTK file. Arrays whose length matches the number of mesh
points become ``POINT_DATA``; arrays matching the number of cells become
``CELL_DATA``:

.. code-block:: python

   writeVtkState("output/ase-{step}.vtk", state, field="phiAse")
   writeVtkState("output/state-{step}.vtk", state,
                   field=("excitationFraction", "relativeStandardError"))

Filenames may contain ``{step}``, ``{time}``, and ``{field}`` placeholders.
Use ``fields={"gain": gain_array}`` to write explicitly named custom arrays.

For custom analysis, consume the NumPy views directly:

.. code-block:: python

   def save_excitation(state, output_dir):
       np.save(output_dir / f"beta-{state.step:06d}.npy", state.excitationFraction)

   simulation.onStep(save_excitation, output_dir)

The former ``vtkWedge`` and ``calcGainFromState`` modules have been removed.
Use ``writeVtkState`` for VTK output and compute application-specific derived
fields explicitly before passing them through ``fields=...``.

Low-level openPMD schema helpers
--------------------------------

The public namespace still exposes schema-building objects such as
``PrimitiveFieldSpec``, ``PointSchema``, ``TriangleSchema``, and
``PrismSchema`` for advanced openPMD tooling. They do not replace
``UnstructuredMesh`` or the material/layout API and are not required for a
normal simulation. Field specifications accept a physical ``Unit`` and expose
conversion helpers after resolution to ``FieldSpec``:

.. code-block:: python

   spec = PrimitiveFieldSpec("temperature", float, unit=units.K)
   field = spec.toFieldSpec(("cell",))
   stored = field.storage_value(300 * units.K)
