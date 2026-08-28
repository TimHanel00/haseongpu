Simulation
==========

``Simulation`` owns every optical component, the current gain medium, initial
and evolving excitation, solver and pump configuration, and time integration.

.. code-block:: python

   simulation = Simulation(
       opticalComponents=[crystal, cladding],
       gainMedium=gainMedium,
       exteriorSurface=exposedSurface,
       initialExcitation=0.0,
       phiASE=phiAse,
       timeIntegrator=FrozenPhiAseRungeKutta4(),
       timeStepSize=2e-5,
       simulationSteps=150,
       prePump=True,
   )
   simulation.addPump(
       pump,
       SurfacePumpInjector(pumpFace),
       relays=(PlanarPumpRelay.retroreflect(returnFace),),
   )

Components outside ``gainMedium`` are passive transport regions. Their
materials may define ``bulkAttenuation`` for volumetric loss; omission means
zero bulk loss. Their ``active`` flag must be false, so their excitation remains
zero. Gain components require an active material and positive
``activeIonDensity``. Components may reference different resolved materials;
each cell retains its material ID and uses that material's spectroscopic and
attenuation data without selecting or averaging between materials.

Physical validation
-------------------

Component volume domains in one simulation must be disjoint. A scalar
``initial_excitation`` covers the complete gain domain. Domain/value mappings
must cover that domain exactly once; gaps, overlap, out-of-range values, and
wrong domain kinds are rejected.

``exteriorSurface`` is an optional user-supplied surface domain. When it is
omitted or ``None``, ``Simulation`` forms the temporary union of all component
domains and uses its ``boundary()``. This excludes internal gain--cladding and
cladding--cladding faces on a shared topology. The inferred union is not stored
as a public ``Simulation.opticalDomain``. Supplying ``exteriorSurface``
overrides the inference. Passive cells receive zero excitation automatically.

During lowering, cells from a shared topology are included once and retain
adjacency. Independent meshes are concatenated as disconnected Tet4 regions
without geometric welding. Domain selections are remapped before excitation,
surface optics, passive-region cell types, pump apertures, or openPMD fields
are emitted.

.. _simulation-callback-lifecycle:

Execution
---------

``step()`` advances the configured ``simulation_steps``. ``step(count)`` uses
an explicit count, and ``runUntil(maxTime)`` advances to a physical time.
Configure at most one of ``simulation_steps`` and ``max_time``. If neither is
set, the active ``PhiASE.ase_steps`` and pump ``pump_steps`` determine the run
length.

Callbacks use lower-camel names:

.. code-block:: python

   simulation.onInit(prepareInitialState)
   simulation.onStep(writeState)
   simulation.beforeStep(updateDebugState)
   simulation.step()
   finalState = simulation.getLastState()

``onInit`` runs once before the first compiled launch. ``onStep`` receives
selected completed ``TimeStepState`` snapshots. ``beforeStep`` is available
only in ``synchronized-debug`` execution with streaming openPMD transport.
Normal ``autonomous`` execution does not synchronize Python between steps.

``output_steps`` selects snapshots; omission selects every completed step.
``output_fields`` accepts ``beta_volume``, ``phi_ase``, ``standard_error``,
``relative_standard_error``, ``total_rays``, ``dndt_ase``, and ``dndt_pump``.
The current synchronized-debug control field is ``beta_volume``.
