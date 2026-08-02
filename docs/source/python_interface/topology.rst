Unstructured mesh topology
==========================

``UnstructuredMesh`` stores Tet4 points, connectivity, adjacency, and volume
and surface domain identity. It deliberately stores no material definitions,
optical behavior, cross sections, or excitation state.

Construction
------------

Create a mesh directly:

.. code-block:: python

   mesh = UnstructuredMesh.from_tetrahedra(
       points, cells,
       volume_domains=[10, 20],
       surface_domains=face_tags,
       volume_domain_names={10: "gain", 20: "window"},
       surface_domain_names={1: "pump_input"},
   )

or load a Tet4 volume:

.. code-block:: python

   mesh = UnstructuredMesh.from_file("crystal.msh")
   mesh = UnstructuredMesh.from_file("crystal.vtk")
   mesh = UnstructuredMesh.from_file("closed_surface.stl", meshSize=0.05)

Gmsh physical names become layout selectors. Numeric physical tags can be used
as selectors as well. ``number_of_points``, ``number_of_cells``,
``cell_connectivity``, ``volume_domain_ids``, ``surface_domain_ids``, and the
neighbor arrays are read-only topology views.

Only Tet4 topology is part of the public simulation API. Material assignment is
performed later with ``Simulation.add_material``.
