Simulation assembly, time, and result quantities
================================================

``Simulation`` composes mesh topology, physical definitions, numerical
solvers, and evolving state. Configuration remains separate so each quantity
has an unambiguous owner.

Construction and time controls
------------------------------

``timeStepSize`` and ``maxTime`` require physical time quantities.
``maxSteps`` is a dimensionless integer limit:

.. code-block:: python

   simulation = Simulation(mesh=mesh, aseSolver=aseSolver, pumpSolver=pumpSolver,
       timeIntegrator=RungeKutta4(), timeStepSize=20 * units.us,
       initialState=InitialState(0.0 * units.one), maxTime=3 * units.ms)

``InitialState`` is the dimensionless upper-state fraction :math:`\beta`, not
an ion density. Material density is configured independently on
``MaterialCondition``.

Register physics with selections owned by the simulation mesh:

.. code-block:: python

   simulation.addMaterial(material, domains=mesh.volume("crystal"))
   simulation.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)
   simulation.addPump(pump, SurfacePumpInjector(mesh.surface("pump_input")))

Registration order determines dense table IDs, not physical precedence.
Overlapping registrations are errors.

Resolve, validate, execute
--------------------------

``resolveProblem()`` resolves selections and validates complete material, exterior
boundary, interface, and initial-state coverage without launching transport:

.. code-block:: python

   problem = simulation.resolveProblem()
   print(problem.cellMaterialId, problem.unsupportedFeatures())

``validateBackend()`` additionally rejects a valid frontend model when the
current adapter cannot execute one of its physical features. ``step(numberOfSteps=1,
pumpSteps=None)`` then transfers the initialized problem once and lets the C++
backend advance the complete requested run. A compiled ``Simulation`` is not a
Python-controlled stepper: a second call would require another backend
initialization and is therefore rejected.
``runUntil(maxTime=...)`` accepts another time quantity.

After initialization, material, boundary, interface, pump, and initialization
callback registrations are frozen. ``currentStep`` is an integer;
``currentTime`` is a time ``Quantity``:

.. code-block:: python

   simulation.step(3)
   print(simulation.currentTime.toValue(units.ms))

Callbacks
---------

``onInit(callback, *args, **kwargs)`` runs once before adapter creation and
receives the ``Simulation``. ``onStep`` runs for each selected native snapshot
and receives a ``TimeStepState``:

.. code-block:: python

   def report(state):
       print(state.step, state.time.toValue(units.us), state.excitationFraction.mean())

   simulation.onStep(report).step(3)

Execution and output contracts
------------------------------

``executionMode="autonomous"`` is the normal performance contract. Python
writes one initialization iteration, the C++ backend owns all time steps, and
only the requested snapshots cross the openPMD boundary. Select one-based
completed-step indices with ``outputSteps`` and fields with ``outputFields``:

.. code-block:: python

   simulation = Simulation(
       # ...
       executionMode="autonomous",
       outputSteps=(40, 150),
       outputFields=("beta_volume", "phi_ase", "relative_standard_error"),
   )
   simulation.step(150)

Omitting ``outputSteps`` emits every completed step. ``autonomousFinal(150)``
is a thin helper returning ``(150,)``; it does not select a different backend
path. Supported output fields are:

* ``beta_volume``
* ``phi_ase``
* ``standard_error``
* ``relative_standard_error``
* ``total_rays``
* ``dndt_ase``
* ``dndt_pump``

``executionMode="synchronized-debug"`` emits every step and waits for Python
before the next one. It requires a streaming openPMD backend. Register
``onControl`` callbacks to return selected control fields:

.. code-block:: python

   simulation = Simulation(
       # ...
       executionMode="synchronized-debug",
       controlFields=("beta_volume",),
   )

   def clamp_beta(state):
       return {"beta_volume": np.minimum(state.excitationFraction, 0.8)}

   simulation.onControl(clamp_beta).step(3)

The initial synchronized-debug contract supports only ``beta_volume``. Static
topology, material, spectra, and launch controls are never resent. This mode is
for diagnostics and interactive control, not normal throughput runs.

State fields and physical meaning
---------------------------------

``TimeStepState`` is a snapshot; its arrays are not a supported mutation path
for native state. The principal fields are:

``time``
   Physical time as a ``Quantity``.

``excitationFraction``
   Dimensionless upper-state fraction on Tet4 cells.

``phiAse``
   ASE photon flux on Tet4 cells, physically
   :math:`\mathrm{m}^{-2}\,\mathrm{s}^{-1}`. The value already includes the
   active-ion-density and fluorescence-lifetime scaling.

``dExcitationDtAse`` and ``dExcitationDtPump``
   Rates of change of the dimensionless excitation fraction, in
   :math:`\mathrm{s}^{-1}`.

``standardError`` and ``relativeStandardError``
   Absolute ASE sampling uncertainty in photon-flux units and dimensionless
   relative uncertainty. See :doc:`uncertainty`.

``totalRays``
   The number of deposited ray visits per Tet4, not the globally launched ray
   budget.

Read a completed snapshot with:

.. code-block:: python

   state = simulation.getLastState()
   print(state.phiAse, state.relativeStandardError)

``getLastState()`` raises before the first completed step.
