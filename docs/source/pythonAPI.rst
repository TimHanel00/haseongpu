Python API Reference
====================

This page is generated from the public Python objects exposed by ``HASEonGPU``.
It is a reference for signatures and members, not a workflow description. Start
with the :doc:`laserPumpCladding tutorial <laserPumpCladding>` for a complete
example and use the :doc:`Python Interface Guide <pythonInterface>` to navigate
the reusable concepts.

Public API
----------

.. currentmodule:: HASEonGPU

Materials and physical graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   Material
   MaterialLibrary
   CrossSectionTable
   OpticalComponent
   Domain
   VolumeTopology
   GainMedium
   Gmsh
   DomainMap
   SurfaceDomainMap
   SurfaceOptics

Legacy planar geometry
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   Grid
   MeshTopology
   vtkWedge

Spectra, pump, and ASE
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   CrossSectionData
   LaserProperties
   SpectralDecomposition
   PumpSpectrum
   PumpAngularDistribution
   UniformPumpProfile
   SuperGaussianPumpProfile
   Pump
   GaussianPump
   SurfacePumpInjector
   PlanarPumpRelay
   integrate_pump_profile
   PhiASE

Simulation and time integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   Simulation
   TimeStepState
   TransportResult
   ExplicitEuler
   ExponentialEuler
   FrozenPhiAseRungeKutta4
   FrozenSourcesRungeKutta4
   Heun
   ImplicitEuler
   Midpoint
   RungeKutta4
   TimeIntegrationSolver

Utilities
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:

   AlpakaBackends
   OpenPmdBackends
   Quantity
   Unit
   backendFlat
   calcGainFromState
   writeParaviewState

The ``units`` namespace contains unit values used to construct ``Quantity``
objects. Each unit carries its openPMD SI scale and unit-dimension tuple.
