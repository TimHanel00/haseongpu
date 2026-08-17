Laser pump and ASE tutorial
===========================

This tutorial constructs a time-dependent Yb:YAG amplifier on an explicit
Tet4 mesh. The model combines a resolved material condition, domain-assigned
surface optics, a finite pump relay, forward ASE transport, and compiled time
integration.

Two launchers describe the same simulation:

* ``example/laserPumpCladdingApi.py`` constructs the graph in Python;
* ``example/laserPumpCladdingYaml.py`` loads
  ``config/laserPumpCladding.yaml``.

The Python and YAML forms are useful for different workflows, but they lower
to the same backend fields and spectra.

Run the example
---------------

Select a compute backend provided by the installed runtime. The YAML example
uses the backend recorded in its configuration:

.. code-block:: console

   python example/laserPumpCladdingYaml.py \
       --config config/laserPumpCladding.yaml \
       --vtk-output-dir output

The Python launcher accepts run-time overrides without changing the physical
construction:

.. code-block:: console

   python example/laserPumpCladdingApi.py \
       --backend Host_Cpu_CpuOmpBlocks \
       --output-steps 50 150 \
       --vtk-output-dir output

Both launchers report completed steps and write selected cell fields. Backend
and openPMD availability depends on the installed runtime; query
``AlpakaBackends.all()`` and ``OpenPmdBackends.all()`` when adapting the
commands to another system.

Resolve the material condition
------------------------------

The material database stores the room-temperature Yb:YAG optical state. The
example resolves that state with the active-ion density and spectral grid used
for this amplifier:

.. literalinclude:: ../../example/laserPumpCladdingApi.py
   :language: python
   :pyobject: laserPumpCladdingMaterial

Temperature, refractive index, fluorescence lifetime, and spectra belong to
the selected material state. ``activeIonDensity`` is a run-specific condition:
it is not fixed by the Yb:YAG database record. The resolved
``Material.crossSections`` supplies both ASE and pump interaction coefficients.
Passive ``bulkAttenuation`` is not required for this active material.

The default spectral resolution is 1000 samples. Resampling happens in the
material layer before the frontend produces the backend ``CrossSectionData``.
See :doc:`python_interface/materials` for material-library semantics and
:doc:`python_interface/spectral_decomposition` for the spectral grid.

Load geometry and select domains
--------------------------------

``VolumeTopology.fromFile`` reads the Tet4 VTK mesh. ``Domain`` then selects
the complete crystal volume and three disjoint exterior regions: the lower
end, upper end, and side surface.

.. literalinclude:: ../../example/laserPumpCladdingApi.py
   :language: python
   :pyobject: laserPumpCladdingComponent

The component binds the material condition to the selected volume. Surface
optics are attached to that component because they describe its boundary, not
mesh connectivity. The end faces use the material refractive index against an
exterior index of 1.0. The side assignment retains the established example
model with matching unit indices.

The returned ``bottom`` and ``top`` domains are also the pump injection and
relay apertures. A single typed surface selection therefore identifies the
same physical face consistently for optics and pump transport.

The historical example name refers to pumping through the cladding geometry,
but this configuration assigns the complete Tet4 volume to the active crystal;
it does not create a separate absorbing cladding volume. A true material
partition uses disjoint gain and passive ``OpticalComponent`` objects. The
compact :doc:`python_interface/materials` example demonstrates that assembly.

Construct the gain medium and solvers
-------------------------------------

``buildSimulation`` assembles the remaining graph:

.. literalinclude:: ../../example/laserPumpCladdingApi.py
   :language: python
   :pyobject: buildSimulation

The ``GainMedium`` gives the crystal component the executable gain role.
``initialExcitation=0.0`` initializes the cell-centered upper-state fraction.
The current backend derives active-ion density, fluorescence lifetime, and
spectra from the component's material when it lowers the graph.

The pump multiplies the configured peak power-density scale of 16,000 by the
numerically integrated super-Gaussian aperture area to obtain
``Pump.total_power``. It launches 50,000 rays at 940 nm for the first 50 outer
steps. ``SurfacePumpInjector`` places the source on the lower face;
``PlanarPumpRelay.retroreflect`` maps rays leaving the upper face back through
the same aperture.

``PhiASE`` performs source-driven forward Monte Carlo transport. The default
run uses adaptive global ray batches, an RSE target of 0.1, and the
surface-reservoir reflection model. ``FrozenPhiAseRungeKutta4`` reuses the
first ASE evaluation within each RK4 outer step while reevaluating pump
transport at every stage.

Run and inspect snapshots
-------------------------

The launcher registers reporting and output callbacks before entering the
compiled loop. ``TimeStepState`` contains the selected completed-step fields,
including excitation, ASE flux, statistical error, and pump/ASE derivatives.

.. code-block:: python

   simulation = buildSimulation(outputSteps=(50, 150))
   simulation.onStep(printState)
   simulation.step()
   finalState = simulation.getLastState()

The example's VTK callback also evaluates the local small-signal gain from the
material-derived cross sections and active-ion density. ``outputSteps`` limits
the snapshots transferred to Python; it does not change the number of compiled
time steps.

Equivalent YAML construction
----------------------------

The YAML file mirrors the same physical graph. Its material section resolves
the HDF5 record before the component refers to it:

.. code-block:: yaml

   schema_version: 3

   materials:
     yb_yag:
       from_hdf5:
         path: ../material_library/data/materials.h5
         key: YbYAG
       temperature: 293.15
       active_ion_density: 2.776e20
       interpolation: exact
       spectral_resolution: 1000

   optical_components:
     crystal:
       domain: amplifier_volume
       material: yb_yag

   gain_media:
     amplifier:
       components: [crystal]

``yb_yag`` is a local reference name. The HDF5 path and ``YbYAG`` key select
the stored record; the remaining values resolve the run-specific condition.
Canonical YAML units for this section are kelvin and ``cm^-3``.

``Simulation.fromYaml`` resolves the complete file:

.. literalinclude:: ../../example/laserPumpCladdingYaml.py
   :language: python
   :pyobject: buildSimulation

Use the YAML form for durable parameter studies and scheduler workflows. Use
the Python form when geometry, material data, or domain composition is produced
programmatically. Named Python materials and other graph objects can also be
injected into ``Simulation.fromYaml`` for mixed workflows.

Model parameters
----------------

The example retains the established physical and numerical values: 293.15 K,
``2.776e20 cm^-3`` active-ion density, 0.941 ms fluorescence lifetime, 1.83
crystal refractive index, 940 nm pump wavelength, 50,000 pump rays, 50 pump
steps, and 150 simulation/ASE steps. The versioned HDF5 record contains the
same absorption and emission data formerly read from four independent text
files.

The estimator, reflection, and population equations are described in
:doc:`theoryAndModel`. Backend selection and transport configuration are
separate from the physical model; see :doc:`backendSelection` and
:doc:`openpmdTransport`.
