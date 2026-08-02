Simulation assembly
===================

``Simulation`` owns composition and evolving state, not physical material
objects or topology.

.. code-block:: python

   simulation = Simulation(
       mesh=mesh,
       ase_solver=ase_solver,
       pump_solver=pump_solver,
       time_integrator=RungeKutta4(),
       time_step_size=1e-5,
       initial_state=InitialState(excitation_fraction=0.0),
       max_steps=100,
   )
   simulation.add_material(material, MaterialLayout("crystal"))
   simulation.add_boundary(boundary, BoundaryLayout("all_exterior"))

``compile()`` creates a backend-neutral ``CompiledProblem`` containing dense
material IDs, boundary IDs, interface IDs, and cell-centred initial excitation.
It validates exact coverage, unknown selectors, overlaps, and missing unlike
material interfaces without invoking transport.

``validate_backend()`` additionally checks the current native capability set.
At present that set is one isotropic active material, Monte Carlo ASE and pump
solvers, and no internal material interfaces or per-material bulk attenuation.
Frontend compilation of multiple materials and Fresnel/transmission interfaces
is supported for inspection and future adapters.

``step(nsteps=1)`` and ``run_until`` execute the supported backend subset.
``on_step`` callbacks receive ``TimeStepState``; its public state views include
``excitation_fraction``, ``d_excitation_dt_ase``, and
``d_excitation_dt_pump``.
