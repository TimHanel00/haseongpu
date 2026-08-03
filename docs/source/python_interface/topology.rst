Unstructured mesh topology and physical scale
==============================================

``UnstructuredMesh`` contains Tet4 coordinates, connectivity, and domain
identity. It deliberately does not own materials, optical boundary behavior,
cross sections, excitation, or pumps.

Coordinates and units
---------------------

Point coordinates are numeric arrays and ``coordinateUnit`` declares their
physical length scale. A point ``[1, 0, 0]`` with ``units.cm`` is one
centimetre from the origin:

.. code-block:: python

   mesh = UnstructuredMesh.fromFile("crystal.msh", coordinateUnit=units.mm)

This unit affects cell volumes, ray path lengths, attenuation, gain, pump
profile sizes, and relay offsets. It is not merely display metadata.

Construct a mesh directly from ``(N, 3)`` points and ``(M, 4)`` point indices:

.. code-block:: python

   mesh = UnstructuredMesh.fromTetrahedra(points, cells,
       volume_domains=cell_tags, surface_domains=face_tags, coordinateUnit=units.cm)

``volume_domains`` has one tag per Tet4. ``surface_domains`` has shape
``(numberOfCells, 4)`` and follows the local Tet4 face order; zero means no
named surface domain. Only tagged exterior faces can be selected by name or
numeric tag.

Supported files
---------------

Load Gmsh, VTK, or closed STL geometry with an explicit coordinate unit:

.. code-block:: python

   mesh = UnstructuredMesh.fromFile("crystal.msh", coordinateUnit=units.mm)
   mesh = UnstructuredMesh.fromFile("legacy.vtk", coordinateUnit=units.cm)

Gmsh physical names become volume and surface domain names. VTK must contain
Tet4 cells. A closed STL surface is tetrahedralized, so its additional keyword
arguments belong to the Gmsh-backed meshing path.

Domain selections
-----------------

Selections are typed objects tied to their originating mesh. Select one or
several physical groups by name or numeric tag:

.. code-block:: python

   gain_cells = mesh.volume("gain")
   launch_faces = mesh.surface("pump_left", "pump_right")

``mesh.exteriorFaces`` selects every exterior face, including faces without a
named physical group:

.. code-block:: python

   simulation.addBoundary(AbsorbingSurface(), domains=mesh.exteriorFaces)

``mesh.boundaryCells`` selects the Tet4 cells touching at least one exterior
face, which is useful for assigning or inspecting a boundary layer:

.. code-block:: python

   print(mesh.boundaryCells.indices)

Use volume selections for ``addMaterial`` and selection-mapped
``InitialState`` values. Use surface selections for ``addBoundary`` and
``SurfacePumpInjector``. Passing a selection from another mesh, or the wrong
selection kind, is an error.

Inspection and validity
-----------------------

The main read-only views are ``points``, ``cellConnectivity``,
``volumeDomainIds``, ``surfaceDomainIds``, ``neighborCells``, and
``neighborLocalFaces``, and ``cellCenters``. Counts are available as
``numberOfPoints`` and ``numberOfCells``; ``coordinateUnit`` reports the
physical coordinate unit.

.. code-block:: python

   print(mesh.numberOfCells, mesh.coordinateUnit.symbol)

Construction rejects non-finite coordinates, invalid or degenerate Tet4
connectivity, and non-manifold faces. Arrays are immutable after construction;
create a new mesh when topology or domain identity must change.
