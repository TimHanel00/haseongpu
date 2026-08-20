Domains, components, and gain media
===================================

Materials describe optical response; mesh implementations provide discrete
connectivity. A ``Domain`` presents a cell-type-independent region interface,
an ``OpticalComponent`` assigns material to a volume domain, and
``GainMedium`` selects the components whose population state participates in
gain transport. See :doc:`materials` for material construction and
persistence.

Domain
------

A ``Domain`` contains either volume cells or local cell faces. Entity kinds
cannot be mixed in arithmetic:

.. code-block:: python

   crystal = Domain.fromGmsh(topology, "crystal", entityKind="volume")
   doped = Domain.fromGmsh(topology, "doped", entityKind="volume")
   passive = crystal - doped
   assembly = doped + passive
   exposedSurface = assembly.boundary()

``Domain.fromTopology(topology)`` selects the complete volume.
``Domain.fromTopology(topology, entityKind="surface")`` selects all exterior
faces. ``Domain.where`` provides the simple ``all_exterior``, ``x_min``,
``x_max``, ``y_min``, ``y_max``, ``z_min``, and ``z_max`` geometric selectors.
Gmsh physical names that occur in more than one dimension require an explicit
entity kind.

Domains may combine several regions and independent mesh bindings. Their set
operations depend on the generic entity and neighbor interface, not on a
particular cell shape. A Hex8 topology can therefore participate in frontend
domain algebra even though the current executable backend accepts Tet4 only.
Empty results remain valid for further set algebra, although physical
consumers reject empty required regions.

``boundary()`` converts a volume-domain union into the oriented surface that
bounds it. Faces between two selected cells are removed, including faces
between separately constructed domains after their union. This is the correct
operation for obtaining the exposed surface of an assembly. Adjacency is known
only within each source topology; independent meshes remain separate optical
bodies and are not welded by coincident coordinates.

OpticalComponent
----------------

An ``OpticalComponent`` accepts one non-empty volume domain and one material:

.. code-block:: python

   from HASEonGPU import OpticalComponent, SurfaceOptics

   gainComponent = OpticalComponent(
       domain=doped,
       material=ybYag,
       name="crystal",
   )
   claddingComponent = OpticalComponent(
       domain=passive,
       material=claddingMaterial,
       name="cladding",
   )
   gainComponent.assignSurfaceOptics(
       pumpFace,
       SurfaceOptics(reflectivity=0.0, n_inside=1.83, n_outside=1.0),
   )

``component.domain`` is the component's volume selection. ``exteriorCells``
returns selected cells touching the
component boundary, including an interface produced by selecting only part of
a shared topology. Surface optics must target a surface domain on the component
boundary.

Components in one simulation must not overlap. Gain and cladding therefore
partition the mesh: the gain component selects the active cells and the
cladding component selects different cells. Their domain union reconstructs
the occupied volume without overlaying two materials on the same cell.

The refractive indices in ``SurfaceOptics`` describe the two sides of the
selected boundary. They can be taken from the component material, but remain
an explicit boundary-model input because the exterior medium may not be part
of the executable graph.

A mirror coating is a surface assignment rather than a bulk material loss:

.. code-block:: python

   mirrorSubstrate.assignSurfaceOptics(
       mirrorSurface,
       SurfaceOptics(reflectivity=0.98, n_inside=1.45, n_outside=1.0),
   )

The substrate may omit ``bulkAttenuation`` when it contributes no volumetric
loss. Rays still traverse its cells, while reflection is applied only at the
selected surface.

GainMedium
----------

``GainMedium`` contains the components executed as gain elements:

.. code-block:: python

   from HASEonGPU import GainMedium

   gainMedium = GainMedium([gainComponent])
   completeGainDomain = gainMedium.domain

Adding a component sets ``component.opticalRole`` to ``"gainElement"``.
Its material must have ``active=True`` and the resolved run condition must
provide a positive ``activeIonDensity``. Conversely, an active material cannot
be placed outside the gain medium. These checks make the material
classification authoritative for whether population excitation is evaluated.
Passive cladding is an ``OpticalComponent`` owned by ``Simulation``, not by
``GainMedium``:

.. code-block:: python

   simulation = Simulation(
       opticalComponents=[gainComponent, claddingComponent],
       gainMedium=gainMedium,
       exteriorSurface=exposedSurface,  # optional
       # solver configuration omitted
   )

``Simulation`` does not expose an aggregate optical domain. If
``exteriorSurface`` is omitted or ``None``, it temporarily unions all component
domains and stores the resulting ``boundary()`` as the exterior surface. This
removes gain--cladding and cladding--cladding interfaces on a shared topology.
Pass an explicit surface to override that inference.

Excitation is defined only over active ``GainMedium`` components; passive cells
are initialized to zero during lowering and do not acquire an excited-state
population. A passive material may define ``bulkAttenuation`` for volumetric
loss. Omitting it selects zero bulk loss.

Backend limits
--------------

Several components may select disjoint cells from one topology, and one
component domain may span several regions. A shared topology retains
gain--cladding adjacency. Independent meshes are concatenated as disconnected
bodies without geometric welding; geometrically touching materials must
therefore use a conforming shared topology.

All gain components currently must reference the same resolved ``Material``
object. The current backend accepts Tet4 ``VolumeTopology`` bindings and
represents all passive cells with one constant coefficient. It rejects both
unsupported cell structures and heterogeneous passive attenuation before
launch. These are backend limits, not restrictions of the domain/component
model.
