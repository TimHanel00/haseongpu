Volume topology
===============

``VolumeTopology`` is the concrete geometry accepted by the current HASEonGPU
backend. It stores an explicit unstructured Tet4 mesh; ASE, pump, and
time-dependent state are cell-centered. It does not store excitation, material
constants, spectra, or boundary optics. Only VTK cell type ``10`` (Tet4) is
supported by this implementation.

All coordinates are metres. Because VTK, Gmsh, and STL files do not encode a
length unit, imported point coordinates are interpreted as metres. Derived
face areas and cell volumes are therefore square metres and cubic metres.

The public ``Domain`` contract is broader. It represents typed mesh regions
through generic cell, face, and neighbor information and does not require a
``VolumeTopology`` instance. This permits frontend composition over other cell
structures. Backend lowering is the point at which unsupported discretizations
are rejected.

Construction
------------

Load gmsh, VTK, or a closed STL surface:

.. code-block:: python

   from HASEonGPU import VolumeTopology

   topology = VolumeTopology.fromFile("crystal.msh")
   topology = VolumeTopology.fromFile("crystal.vtk")
   topology = VolumeTopology.fromFile("crystal.stl", meshSize=0.0005)

Use ``format=`` when a filename has a non-standard extension. ``fromVtk``
expects an ASCII VTK unstructured grid. ``fromStl`` uses gmsh to tetrahedralize
a closed three-dimensional surface and warns that HASEonGPU does not perform a
complete mesh-validity proof.

Construct a topology directly when another mesher already provides arrays:

.. code-block:: python

   topology = VolumeTopology.fromTetrahedra(
       points,                 # (numberOfPoints, 3)
       cellPointIndices,       # (numberOfCells, 4)
       cellDomains=cell_ids,   # optional (numberOfCells,)
       faceBoundaries=faces,   # optional (numberOfCells, 4)
   )

Derived geometry
----------------

Construction derives and validates the arrays needed by transport:

``cellPointIndices``
   Four point indices per cell.

``facePointIndices``
   Three point indices for each of the four local faces.

``neighborCells`` and ``neighborLocalFaces``
   The adjacent cell and its matching local face, or a negative value at the
   exterior boundary.

``cellCenters`` and ``cellVolumes``
   Cell geometry used for state placement, source weighting, pump-rate
   normalization, and volume-weighted post-processing.

``faceCenters``, ``faceNormals``, and ``faceAreas``
   Oriented face geometry used by boundary optics and pump injection.

The main size queries are ``numberOfPoints``, ``numberOfCells``,
``numberOfFacesPerCell``, and ``numberOfSamplePoints``. ``samplePoints`` equals
the cell centers in an explicit volume topology.

Geometry labels and physical domains
------------------------------------

Domains are positive integer labels. Cell domains identify volume regions;
surface domains identify faces used by pump injection, relays, or optical
boundaries. gmsh physical names are retained and can be used instead of numeric
tags. A label is geometry metadata. ``Domain.fromGmsh`` turns a label into a
typed volume or surface selection before a component, pump injector, relay, or
boundary model gives it physical meaning.

.. code-block:: python

   topology = (
       topology
       .withCellDomains(where="all", domain=1, name="gain")
       .withSurfaceDomains([
           {"where": "z_min", "domain": 10, "name": "pump_input"},
           {"where": "z_max", "domain": 11, "name": "pump_output"},
       ])
   )

``withCellDomains`` accepts cell indices, ``where="all"``, and gmsh physical
names or tags. ``withSurfaceDomains`` additionally accepts face indices,
``z_min``, ``z_max``, and all exterior faces. It rejects internal faces unless
``allowInternal=True`` is explicit. Both methods return a copied topology, so
the input object remains unchanged.

The low-level maps remain available for inspecting imported labels:

.. code-block:: python

   entryId = topology.surfaceDomainMap().resolve("pump_input")

For physical construction, resolve a typed domain and assign boundary optics
to its component:

.. code-block:: python

   from HASEonGPU import Domain, OpticalComponent, SurfaceOptics

   crystalVolume = Domain.fromGmsh(topology, "gain", entityKind="volume")
   pumpInput = Domain.fromGmsh(topology, "pump_input", entityKind="surface")
   crystal = OpticalComponent(domain=crystalVolume, material=material)
   crystal.assignSurfaceOptics(
       pumpInput,
       SurfaceOptics(
           reflectivity=0.0,
           n_inside=material.refractiveIndex,
           n_outside=1.0,
       ),
   )

When several materials partition one topology, form the occupied volume before
extracting its surface:

.. code-block:: python

   occupiedVolume = gainDomain + claddingDomain
   exposedSurface = occupiedVolume.boundary()

``boundary()`` retains a selected cell face only when its neighbor is outside
the volume union. Gain--cladding and cladding--cladding interfaces are therefore
not mistaken for exposed faces. This calculation uses topological adjacency,
not coordinate proximity; independently meshed touching components must be
remeshed or composed into a conforming topology first.

``Simulation`` performs this union-and-boundary operation over all optical
components when ``exteriorSurface`` is omitted or ``None``. Pass an explicit
surface when the model requires a different exterior selection. The temporary
occupied volume is not exposed as ``Simulation.opticalDomain``.

See :doc:`gain_medium` for ``SurfaceOptics`` syntax and
:ref:`ase-surface-reflections` for the implemented boundary physics.

VTK geometry input
------------------

``VolumeTopology.fromVtk`` reads an ASCII VTK unstructured grid and constructs
the Tet4 topology. Physical arrays in a legacy VTK file are not converted into
the redesigned ``Material`` or ``GainMedium`` automatically:

.. code-block:: python

   topology = VolumeTopology.fromVtk("geometry.vtk")

Assign materials, excitation, and boundary optics explicitly after loading the
topology. This prevents a geometry file from silently becoming the authority
for run-specific material state.

Legacy extruded topology
------------------------

``Grid`` and ``MeshTopology`` remain available for compatibility with planar
triangle meshes extruded into wedge layers. They expose legacy queries such as
``numberOfTriangles``, ``numberOfLevels``, and ``numberOfPrisms``. New
forward-volume simulations should use ``VolumeTopology``; VTK Tet4 input is
intentionally rejected by ``MeshTopology.fromVtk``.
