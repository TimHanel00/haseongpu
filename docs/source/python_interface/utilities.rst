Utilities
=========

Time integration
----------------

The public time integrators are lightweight descriptors for compiled backend
algorithms: ``ExplicitEuler``, ``Heun``, ``Midpoint``, ``RungeKutta4``,
``FrozenPhiAseRungeKutta4``, ``ImplicitEuler``, and ``ExponentialEuler``.

.. code-block:: python

<<<<<<< HEAD
   from HASEonGPU import (
       ExplicitEuler,
       Heun,
       Midpoint,
       RungeKutta4,
       FrozenPhiAseRungeKutta4,
       ImplicitEuler,
       ExponentialEuler,
   )

Available solvers:

* ``ExplicitEuler()``
* ``Heun()``
* ``Midpoint()``
* ``RungeKutta4()``
* ``FrozenPhiAseRungeKutta4()``: reuses one ASE evaluation across RK4 stages.
* ``ImplicitEuler(iterations=8, tolerance=1e-10)``
* ``ExponentialEuler()``

These objects are lightweight descriptors. Python serializes their ``name`` to
the openPMD run-control record and the compiled C++/Alpaka backend performs the
actual time integration. You can also pass one of the names directly as a
string.

Custom Python time integrators are not supported by compiled simulation runs.

VTK Export
----------

``vtkWedge`` writes point or cell data on the wedge mesh to a legacy ASCII
VTK file. In a ``Simulation.on_step`` callback, pass the ``TimeStepState`` to
``vtkWedge``; the state carries the static topology and the dynamic arrays.

Callback use:

.. code-block:: python

   def write_vtk(state, output_dir, cladding_absorption):
       vtkWedge(
           output_dir / "fields_{step:03d}.vtk",
           state,
           fields={
               "betaCells": state.beta_cells,
               "phiASE": state.phi_ase,
               "dndtAse": state.dndt_ase,
               "cladAbs": state.phi_ase * cladding_absorption,
           },
       )

   simulation.on_step(write_vtk, output_dir, 5.5)

Direct use after one step:

.. code-block:: python

   simulation.step()
   state = simulation.get_last_state()
   vtkWedge("phi.vtk", state)
   vtkWedge("fields.vtk", state, field=["phiAse", "dndtAse"])
   vtkWedge("named.vtk", state, field={"phi": "phiAse", "dn": "dndtAse"})

For standalone array exports outside a simulation state, pass ``geometry`` as a
``GainMedium`` or ``MeshTopology``:

.. code-block:: python

   vtkWedge("fields.vtk", geometry=medium, fields={"phi": phi, "dn": dndt})

The older callback-factory form is still accepted and can use ``every`` to
reduce output frequency:

.. code-block:: python

   simulation.on_step(vtkWedge("phi_{step:03d}.vtk", medium, every=10))

For new code, prefer an explicit callback when output frequency or derived
fields are needed:

.. code-block:: python

   def write_every_tenth(state, output_dir):
       if state.step % 10 == 0:
           vtkWedge(output_dir / "phi_{step:03d}.vtk", state)

   simulation.on_step(write_every_tenth, output_dir)

The data shape must match either:

* point data: ``(numberOfPoints, numberOfLevels)``
* cell data: ``(numberOfTriangles, numberOfLevels - 1)``

Gain Field Export
-----------------

``calcGainFromState`` calculates small-signal laser gain from a
``TimeStepState`` and returns a point-shaped array that can be written directly
with ``vtkWedge``:

.. code-block:: python

   vtkWedge(
       output_path,
       state,
       fields={
           "gain": calcGainFromState(state, spectra, nTot),
       },
   )


Backend Names
=======
   from HASEonGPU import FrozenPhiAseRungeKutta4
   integrator = FrozenPhiAseRungeKutta4()

Backend names
>>>>>>> 0a6b6680 (Introduce PICMI-aligned material and mesh API)
-------------

``AlpakaBackends.all()`` lists compute backends available in the installed
runtime:

.. code-block:: python

   from HASEonGPU import AlpakaBackends, MonteCarloASESolver

   backend = AlpakaBackends.all()[0]
   ase_solver = MonteCarloASESolver(backend=backend)

See :doc:`../backendSelection` for compute versus openPMD backend selection.

State export
------------

``writeParaviewState`` remains available for backend state export. Callback
consumers can also use the NumPy arrays on ``TimeStepState`` directly. The
public Tet4 views are ``excitation_fraction``, ``d_excitation_dt_ase``, and
``d_excitation_dt_pump``.
