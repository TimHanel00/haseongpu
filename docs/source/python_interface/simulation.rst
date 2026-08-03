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
pumpSteps=None)`` then advances a fixed number of outer time steps.
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
receives the ``Simulation``. ``onStep`` runs after each completed native step
and receives a ``TimeStepState``:

.. code-block:: python

   def report(state):
       print(state.step, state.time.toValue(units.us), state.excitationFraction.mean())

   simulation.onStep(report).step(3)

State fields and physical meaning
---------------------------------

``TimeStepState`` is a snapshot; its arrays are not a supported mutation path
for native state. The principal fields are:

``time``
   Physical time as a ``Quantity``.

``sampledExcitationFraction`` and ``excitationFraction``
   Dimensionless upper-state fractions on the backend's sample points and
   Tet4 cells, respectively.

``phiAse`` and ``volumePhiAse``
   ASE photon flux on sample points and Tet4 cells, physically
   :math:`\mathrm{m}^{-2}\,\mathrm{s}^{-1}`. The value already includes the
   active-ion-density and fluorescence-lifetime scaling.

``sampledDExcitationDtAse``, ``volumeDExcitationDtAse``, and ``dExcitationDtPump``
   Rates of change of the dimensionless excitation fraction, in
   :math:`\mathrm{s}^{-1}`.

``volumeStandardError`` and ``volumeRelativeStandardError``
   Absolute ASE sampling uncertainty in photon-flux units and dimensionless
   relative uncertainty. See :doc:`uncertainty`.

``volumeTotalRays``
   The number of deposited ray visits per Tet4, not the globally launched ray
   budget.

Read a completed snapshot with:

.. code-block:: python

   state = simulation.getLastState()
   print(state.volumePhiAse, state.volumeRelativeStandardError)

``getLastState()`` raises before the first completed step.
