Migrating from the previous frontend
====================================

The composition frontend is a breaking API change. The old aggregate objects
were removed from the public ``HASEonGPU`` namespace instead of being retained
as ambiguous aliases. This page maps each former responsibility to its new
owner.

Object mapping
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Previous public object or field
     - Replacement
     - Important change
   * - ``MeshTopology`` or ``VolumeTopology``
     - ``UnstructuredMesh``
     - Public geometry is explicit Tet4 topology with immutable views.
   * - ``GainMedium``
     - ``Material`` + ``MaterialCondition`` + mesh selections + ``InitialState``
     - Material physics and evolving excitation no longer live in the mesh.
   * - ``SpectralDecomposition`` or ``CrossSectionData``
     - ``CrossSectionTable``
     - One wavelength grid with explicit length and area units.
   * - ``PhiASE``
     - ``MonteCarloASESolver``
     - Constructor keywords use lower camel case, for example ``minRays``.
   * - old ``Simulation(gain_medium=..., phiASE=...)``
     - ``Simulation(mesh=..., aseSolver=..., ...)``
     - Physics is registered after construction with ``addMaterial``,
       ``addBoundary``, ``addInterface``, and ``addPump``.
   * - pump ``crossSections``
     - the selected material's ``CrossSectionTable``
     - A pump describes incident light; interaction data belongs to material.
   * - ``TransportResult``
     - ``TimeStepState`` from ``onStep`` or ``getLastState()``
     - Completed-step state is the supported public result surface.

Units
-----

The public composition API uses explicit unit-bearing values. Attach the unit
present in the source data instead of applying a manual SI conversion:

.. code-block:: python

   wavelength = 940 * units.nm
   density = 2.76e20 / units.cm**3

``CrossSectionTable.fromTextDirectory(path)`` reads the historical
``lambda_a.txt``, ``sigma_a.txt``, ``lambda_e.txt``, and ``sigma_e.txt`` files
and declares their historical ``nm`` and ``cm^2`` units. It emits a warning;
new material databases use the versioned HDF5 representation.

Before and after
----------------

The former aggregate setup assigned arrays directly to a ``GainMedium`` and
passed that object into ``Simulation``. The new equivalent is assembled by
responsibility:

.. code-block:: python

   mesh = UnstructuredMesh.fromFile("crystal.msh", coordinateUnit=units.mm)
   spectra = CrossSectionTable.fromTextDirectory("material-data")
   yag = Material("Yb:YAG").addState(
       temperature=None,
       refractiveIndex=1.82,
       fluorescenceLifetime=941 * units.us,
       crossSections=spectra,
       metadata={"temperature_status": "not documented by source"},
   )
   material = yag.at(activeIonDensity=2.76e20 / units.cm**3)

   simulation = Simulation(
       mesh=mesh,
       aseSolver=MonteCarloASESolver(
           minRays=100_000,
           maxRays=1_000_000,
           backend="Host_Cpu_CpuSerial",
       ),
       pumpSolver=MonteCarloPumpSolver(rayCount=100_000),
       timeIntegrator=RungeKutta4(),
       timeStepSize=10 * units.us,
       initialState=InitialState(0.0 * units.one),
   )
   simulation.addMaterial(material, domains=mesh.volume("crystal"))
   simulation.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)

Retired frontend structures and compatibility aliases are not supported.
Do not import them from ``pyInclude``; migrate to the public objects listed in
:doc:`../pythonAPI`.
