Python API Reference
====================

The public API follows a PICMI-style composition: topology, physical material
models, selections, solver descriptors, and state are independent objects composed
by ``Simulation``.

.. currentmodule:: HASEonGPU

Mesh and materials
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   UnstructuredMesh
   MeshSelection
   CrossSectionTable
   Material
   MaterialState
   MaterialCondition
   MaterialLibrary
   Unit
   Quantity

``materialLibrary`` is the preloaded built-in ``MaterialLibrary``. Its
``YbYAG`` cross sections were recorded at room temperature, represented as
``293.15 * units.K``. Select it with ``materialLibrary["YbYAG"]``.

Material data warnings
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LegacyMaterialTextWarning
   TemperatureInterpolationWarning

Boundaries and interfaces
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ExteriorBoundaryModel
   AbsorbingSurface
   ConstantReflectivitySurface
   MaterialInterfaceModel
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
   integratePumpProfile

Simulation and state
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   InitialState
   Simulation
   ResolvedProblem
   BackendCapabilities
   currentBackendCapabilities
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
   writeVtkState

Advanced openPMD schema helpers
-------------------------------

These helpers are public for applications that build HASE-compatible openPMD
records directly. Their field and attribute specifications accept ``Unit``
objects and retain ``unitSI`` and ``unitDimension`` metadata. Normal
``Simulation`` assembly does not need them.

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
