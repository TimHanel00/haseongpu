Python Interface Guide
======================

The Python interface separates **what the device is made of** from **how it is
meshed**. This follows PICMI's composition pattern and avoids storing material
physics or evolving state inside the mesh.

A simulation is assembled in this order:

#. create a Tet4 :doc:`mesh <python_interface/topology>`;
#. define reusable material physics and run-specific material instances;
#. attach each instance to volume domains with ``MaterialLayout``;
#. attach boundary and internal-interface models to their layouts;
#. choose ASE, pump, and time-integration solvers;
#. supply ``InitialState`` and register physical pumps;
#. call ``compile()``, ``validate_backend()``, or ``step()``.

.. important::

   The frontend can compile multiple materials and explicit
   ``PerfectTransmission`` or ``FresnelInterface`` models today. The current
   C++/openPMD 0.1 backend adapter supports only one isotropic active material
   and no internal material interface. It rejects unsupported configurations
   before transport. Fresnel reflection/refraction and transmission between
   mesh domains are therefore not implemented in the backend yet.

   ``ASESolver`` and ``PumpSolver`` are extensible roles. The current adapter
   implements only ``MonteCarloASESolver`` and ``MonteCarloPumpSolver``; a
   different descriptor can be composed in Python but is rejected at backend
   validation until an adapter is provided.

Concept Pages
-------------

.. toctree::
   :maxdepth: 2

   python_interface/topology
   python_interface/gain_medium
   python_interface/spectral_decomposition
   python_interface/pump_properties
   python_interface/phi_ase
   python_interface/simulation
   python_interface/utilities

One-material example
--------------------

All public physical values use SI units.

.. code-block:: python

   from HASEonGPU import *

   mesh = UnstructuredMesh.from_file("crystal.msh")
   spectra = CrossSectionTable.monochromatic(
       wavelength=1030e-9, absorption=1.2e-25, emission=2.48e-24
   )
   yag = MaterialDefinition(
       "Yb:YAG", refractive_index=1.82,
       fluorescence_lifetime=941e-6, cross_sections=spectra,
   )
   crystal = MaterialInstance(yag, active_ion_density=2.76e26)

   simulation = Simulation(
       mesh=mesh,
       ase_solver=MonteCarloASESolver(backend="Host_Cpu_CpuSerial"),
       pump_solver=MonteCarloPumpSolver(ray_count=100_000),
       time_integrator=RungeKutta4(),
       time_step_size=1e-5,
       initial_state=InitialState(0.0),
   )
   simulation.add_material(crystal, MaterialLayout("crystal"))
   simulation.add_boundary(
       ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior")
   )
   simulation.add_pump(
       Pump(total_power=16e3, spectrum=PumpSpectrum.monochromatic(940e-9)),
       SurfacePumpInjector("pump_input"),
   )
   simulation.step(3)

See ``example/minimalExampleNewInterface.py`` for a self-contained mesh and
``example/gmshMinimalExample.py`` for named gmsh volume/surface domains.
