Simulation assembly and execution
=================================

``Simulation`` owns composition and evolving state. It references topology and
physical definitions but does not move those responsibilities into one
aggregate object.

Construction and registration
-----------------------------

.. code-block:: python

   simulation = Simulation(
       mesh=mesh,
       ase_solver=ase_solver,
       pump_solver=pump_solver,
       time_integrator=RungeKutta4(),
       time_step_size=1e-5,
       initial_state=InitialState(0.0),
       max_steps=100,
       max_time=1e-3,
       enable_ase=True,
       pre_pump=False,
   )
   simulation.add_material(material, MaterialLayout("crystal"))
   simulation.add_boundary(boundary, BoundaryLayout("all_exterior"))
   simulation.add_pump(pump, SurfacePumpInjector("pump_input"))

The ``add_*`` methods return the simulation, so chaining is possible, but
separate calls are often clearer. Registration order determines dense table
IDs, not physical precedence. Overlapping layouts are errors.

Compile, validate, execute
--------------------------

``compile()`` returns ``CompiledProblem`` without launching the backend. Its
main tables are:

* ``materials`` and ``cell_material_id``;
* ``boundaries`` and ``face_boundary_id``;
* ``interfaces`` and ``face_interface_id``;
* ``initial_excitation_fraction``.

This is the right stage for configuration tools and tests to inspect domain
resolution. ``compiled.unsupported_features()`` reports the difference between
the problem and ``CURRENT_BACKEND_CAPABILITIES``.

``validate_backend()`` compiles and then raises ``NotImplementedError`` for an
unsupported physical feature or solver role. ``step(nsteps=1,
pump_steps=None)`` performs that validation and executes a fixed number of
steps. ``run_until(max_time=None)`` uses the argument or the constructor's
``max_time``. The current adapter requires at least one registered pump.

Execution initializes the private native/openPMD adapter only once. After that
point, physical registrations and initialization callbacks cannot be changed.
``current_step`` and ``current_time`` report completed progress.

Callbacks and state
-------------------

``on_init(callback, *args, **kwargs)`` runs once just before adapter creation.
It receives the public ``Simulation``. ``on_step`` runs after each completed
native step and receives a ``TimeStepState``:

.. code-block:: python

   def report(state, label):
       print(label, state.step, state.time, state.excitation_fraction.mean())

   simulation.on_step(report, "amplifier")
   simulation.step(3)
   final = simulation.get_last_state()

For public Tet4 runs, the principal views are ``excitation_fraction``,
``d_excitation_dt_ase``, and ``d_excitation_dt_pump``. ``phi_ase`` exposes the
ASE flux when enabled. Lower-level compatibility names remain on
``TimeStepState`` for the private adapter, but new callback code should use the
snake-case public views.

``get_last_state()`` raises before the first completed step. The arrays are
snapshots from that completed step; changing them is not a supported way to
modify native simulation state.
