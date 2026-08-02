Unstructured mesh topology
==========================

``UnstructuredMesh`` contains Tet4 geometry and domain identity only. It does
not own materials, optical boundary behavior, cross sections, or excitation.

Construction
------------

Create a mesh directly from ``(N, 3)`` points and ``(M, 4)`` point indices:

.. code-block:: python

   mesh = UnstructuredMesh.from_tetrahedra(
       points,
       cell_connectivity,
       volume_domains=[10, 20],
       surface_domains=face_tags,
       volume_domain_names={10: "gain", 20: "cap"},
       surface_domain_names={1: "pump_input", 2: "outer"},
   )

``volume_domains`` has one tag per cell. ``surface_domains`` has shape
``(number_of_cells, 4)`` and follows the mesh's local Tet4 face order; zero
means no named surface domain. Only exterior tagged faces are eligible for a
``BoundaryLayout`` or ``SurfacePumpInjector``.

Load an existing Tet4 volume with:

.. code-block:: python

   mesh = UnstructuredMesh.from_file("crystal.msh")
   mesh = UnstructuredMesh.from_file("crystal.vtk")
   mesh = UnstructuredMesh.from_file("closed_surface.stl", meshSize=0.05)

Gmsh physical names become volume and surface selectors. VTK must contain Tet4
cells. A closed STL surface is tetrahedralized, so its optional keyword
arguments depend on the Gmsh-backed meshing path.

Selectors and layouts
---------------------

Layouts accept a numeric tag, a physical name, or a tuple of either:

.. code-block:: python

   MaterialLayout("gain")
   MaterialLayout((10, 20))
   BoundaryLayout(("pump_input", "outer"))

``MaterialLayout("all")`` selects every cell and
``BoundaryLayout("all_exterior")`` selects every exterior face. These special
selectors cannot be combined with other selectors. Unknown or ambiguous names
are reported by ``Simulation.compile()``.

Inspection and validity
-----------------------

The main read-only views are ``points``, ``cell_connectivity``,
``volume_domain_ids``, ``surface_domain_ids``, ``neighbor_cells``, and
``neighbor_local_faces``. Counts are available as ``number_of_points`` and
``number_of_cells``. Domain-name dictionaries are exposed by
``volume_domain_names`` and ``surface_domain_names``.

Construction rejects non-finite coordinates, invalid or degenerate Tet4
connectivity, and non-manifold faces. Arrays are immutable after construction;
create a new mesh when topology or domain identity must change.
