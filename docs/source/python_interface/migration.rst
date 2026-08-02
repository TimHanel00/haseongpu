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
     - ``MaterialDefinition`` + ``MaterialInstance`` + layouts + ``InitialState``
     - Material physics and evolving excitation no longer live in the mesh.
   * - ``SpectralDecomposition`` or ``CrossSectionData``
     - ``CrossSectionTable``
     - One wavelength grid, expressed in metres and square metres.
   * - ``PhiASE``
     - ``MonteCarloASESolver``
     - Constructor keywords are snake case, for example ``min_rays``.
   * - old ``Simulation(gain_medium=..., phi_ase=...)``
     - ``Simulation(mesh=..., ase_solver=..., ...)``
     - Physics is registered after construction with ``add_*`` methods.
   * - pump ``cross_sections``
     - the selected material's ``CrossSectionTable``
     - A pump describes incident light; interaction data belongs to material.
   * - ``TransportResult``
     - ``TimeStepState`` from ``on_step`` or ``get_last_state()``
     - Completed-step state is the supported public result surface.

Units
-----

The public composition API uses SI values. When porting historical setup data:

* wavelengths in nanometres are multiplied by ``1e-9``;
* cross sections in ``cm^2`` are multiplied by ``1e-4``;
* active-ion density in ``cm^-3`` is multiplied by ``1e6``;
* attenuation in ``cm^-1`` is multiplied by ``100``.

``CrossSectionTable.from_directory(path)`` reads the historical
``lambda_a.txt``, ``sigma_a.txt``, ``lambda_e.txt``, and ``sigma_e.txt`` files
and performs the wavelength/cross-section conversions.

Before and after
----------------

The former aggregate setup assigned arrays directly to a ``GainMedium`` and
passed that object into ``Simulation``. The new equivalent is assembled by
responsibility:

.. code-block:: python

   mesh = UnstructuredMesh.from_file("crystal.msh")
   spectra = CrossSectionTable.from_directory("material-data")
   definition = MaterialDefinition(
       "Yb:YAG",
       refractive_index=1.82,
       fluorescence_lifetime=941e-6,
       cross_sections=spectra,
   )
   material = MaterialInstance(definition, active_ion_density=2.76e26)

   simulation = Simulation(
       mesh=mesh,
       ase_solver=MonteCarloASESolver(
           min_rays=100_000,
           max_rays=1_000_000,
           backend="Host_Cpu_CpuSerial",
       ),
       pump_solver=MonteCarloPumpSolver(ray_count=100_000),
       time_integrator=RungeKutta4(),
       time_step_size=1e-5,
       initial_state=InitialState(0.0),
   )
   simulation.add_material(material, MaterialLayout("crystal"))
   simulation.add_boundary(
       ExteriorBoundary(AbsorbingSurface()),
       BoundaryLayout("all_exterior"),
   )

Do not import retired objects from ``pyInclude`` to migrate an application.
That path exists only for repository-owned frozen regression fixtures and may
change without notice. Use :doc:`../pythonAPI` to verify what is public.
