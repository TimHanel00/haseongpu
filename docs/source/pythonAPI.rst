Python API Reference
====================

The public API follows a PICMI-style composition: topology, physical material
models, layouts, solver descriptors, and state are independent objects composed
by ``Simulation``.

.. currentmodule:: HASEonGPU

Mesh and materials
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   UnstructuredMesh
   CrossSectionTable
   MaterialDefinition
   MaterialInstance
   MaterialLayout

Boundaries and interfaces
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ExteriorBoundary
   ExteriorBoundaryModel
   BoundaryLayout
   AbsorbingSurface
   ConstantReflectivitySurface
   MaterialInterface
   MaterialInterfaceModel
   MaterialInterfaceLayout
   PerfectTransmission
   FresnelInterface

Pumps and solvers
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   PumpSpectrum
   PumpAngularDistribution
   UniformPumpProfile
   SuperGaussianPumpProfile
   Pump
   GaussianPump
   SurfacePumpInjector
   PlanarPumpRelay
   ASESolver
   MonteCarloASESolver
   PumpSolver
   MonteCarloPumpSolver
   integrate_pump_profile

Simulation and state
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   InitialState
   Simulation
   CompiledProblem
   BackendCapabilities
   CURRENT_BACKEND_CAPABILITIES
   TimeStepState
   ExplicitEuler
   ExponentialEuler
   FrozenPhiAseRungeKutta4
   Heun
   ImplicitEuler
   Midpoint
   RungeKutta4
   TimeIntegrationSolver

Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   AlpakaBackends
   writeParaviewState

Advanced openPMD schema helpers
-------------------------------

These helpers are public for applications that build HASE-compatible openPMD
records directly. Normal ``Simulation`` assembly does not need them.

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseGroup
   BaseSchema
   GroupFieldSpec
   OpenPmdBackends
   PointSchema
   PrimitiveFieldSpec
   PrismSchema
   TriangleSchema
   backendFlat
