# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Topology-independent regions and their optical material assignments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from material_library import Material
from .geometry import SurfaceOptics


VOLUME = "volume"
SURFACE = "surface"


def _numberOfCells(topology):
    try:
        numberOfCells = int(topology.numberOfCells)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("a volume topology must provide an integer numberOfCells") from exc
    if numberOfCells < 0:
        raise ValueError("topology numberOfCells must be non-negative")
    return numberOfCells


def _neighborCells(topology):
    try:
        neighbors = np.asarray(topology.neighborCells)
    except AttributeError as exc:
        raise TypeError("a surface-capable topology must provide neighborCells") from exc
    expectedCells = _numberOfCells(topology)
    if neighbors.ndim != 2 or neighbors.shape[0] != expectedCells:
        raise ValueError(
            "topology neighborCells must have shape "
            f"(numberOfCells, numberOfFacesPerCell), got {neighbors.shape}"
        )
    return neighbors


def _entity_shape(topology, entityKind):
    if entityKind == VOLUME:
        return (_numberOfCells(topology),)
    if entityKind == SURFACE:
        return _neighborCells(topology).shape
    raise ValueError("entityKind must be 'volume' or 'surface'")


class Domain:
    """Immutable typed region over one or more mesh discretizations.

    A domain contains either volume cells or local cell faces. Its public set
    algebra is independent of the concrete cell type; each internal binding
    retains the source topology needed to resolve the selected entities during
    lowering. Domains from the same topology share adjacency, while domains
    from independent topologies remain disconnected.

    Parameters
    ----------
    value
        Existing domain or iterable of compatible domains to combine.
    entityKind
        ``"volume"`` or ``"surface"``. Required for an empty domain and for
        direct topology/mask construction.
    topology, mask
        Internal discretization binding and boolean entity selection. The
        topology must implement the generic attributes required by the entity
        kind, but need not be a Tet4 ``VolumeTopology``.
    """

    __slots__ = ("entityKind", "_shards")

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("Domain objects are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, value=None, *, entityKind=None, topology=None, mask=None):
        if isinstance(value, Domain):
            if entityKind is not None and entityKind != value.entityKind:
                raise TypeError("cannot change a Domain's entity kind")
            self.entityKind = value.entityKind
            self._shards = value._shards
            return

        if topology is not None or mask is not None:
            if topology is None or mask is None:
                raise TypeError("topology and mask must be provided together")
            if entityKind not in {VOLUME, SURFACE}:
                raise ValueError("entityKind must be 'volume' or 'surface'")
            parsed = self._readMask(topology, mask, entityKind)
            self.entityKind = entityKind
            self._shards = ((topology, parsed),) if np.any(parsed) else ()
            return

        values = (
            ()
            if value is None
            else tuple(value if isinstance(value, Iterable) else (value,))
        )
        if not values:
            if entityKind not in {VOLUME, SURFACE}:
                raise ValueError("an empty Domain requires entityKind")
            self.entityKind = entityKind
            self._shards = ()
            return
        result = Domain(values[0])
        for item in values[1:]:
            result = result + Domain(item)
        if entityKind is not None and entityKind != result.entityKind:
            raise TypeError("domain collection contains the wrong entity kind")
        self.entityKind = result.entityKind
        self._shards = result._shards

    @staticmethod
    def _readMask(topology, mask, entityKind):
        expected = _entity_shape(topology, entityKind)
        parsed = np.asarray(mask, dtype=bool)
        if parsed.shape != expected:
            raise ValueError(
                f"{entityKind} domain mask must have shape {expected}, got {parsed.shape}"
            )
        parsed = parsed.copy()
        parsed.flags.writeable = False
        return parsed

    @classmethod
    def fromTopology(cls, topology, *, entityKind=VOLUME):
        """Select every volume cell or every exterior face of a topology.

        Parameters
        ----------
        topology
            Any topology implementing ``numberOfCells`` and, for a surface
            domain, ``neighborCells``.
        entityKind
            ``"volume"`` selects all cells. ``"surface"`` selects faces
            without a neighboring cell.

        Returns
        -------
        Domain
            A topology-bound region with a cell-type-independent interface.
        """
        mask = np.ones(_entity_shape(topology, entityKind), dtype=bool)
        if entityKind == SURFACE:
            mask &= _neighborCells(topology) < 0
        return cls(entityKind=entityKind, topology=topology, mask=mask)

    @classmethod
    def fromGmsh(cls, topology, physicalGroup, *, entityKind=None):
        """Select a preserved Gmsh physical group from a topology.

        The topology may use any cell representation as long as it exposes
        the imported ``cellDomains``, ``faceBoundaries``, and corresponding
        name mappings.
        """
        candidates = []
        for kind, names, values in (
            (VOLUME, topology.cellDomainNames, topology.cellDomains),
            (SURFACE, topology.surfaceDomainNames, topology.faceBoundaries),
        ):
            if entityKind is not None and kind != entityKind:
                continue
            if isinstance(physicalGroup, str):
                identifiers = [int(tag) for tag, name in names.items() if name == physicalGroup]
            else:
                identifier = int(physicalGroup)
                identifiers = (
                    [identifier]
                    if identifier in set(np.asarray(values).reshape(-1))
                    else []
                )
            for identifier in identifiers:
                candidates.append((kind, np.asarray(values) == identifier))
        if not candidates:
            raise KeyError(f"unknown Gmsh physical group {physicalGroup!r}")
        kinds = {kind for kind, _mask in candidates}
        if len(candidates) != 1 or len(kinds) != 1:
            raise ValueError("ambiguous Gmsh physical group; specify entityKind")
        kind, selected = candidates[0]
        return cls(entityKind=kind, topology=topology, mask=selected)

    @classmethod
    def where(cls, topology, selector, *, entityKind=SURFACE):
        """Select a geometric extremum without creating mesh labels.

        Surface selectors require topology-provided ``neighborCells`` and
        ``faceCenters`` arrays. Volume selection currently supports ``"all"``.
        """
        if entityKind == VOLUME:
            if selector != "all":
                raise ValueError("volume Domain.where only supports the 'all' selector")
            return cls.fromTopology(topology, entityKind=VOLUME)
        if entityKind != SURFACE:
            raise ValueError("entityKind must be 'volume' or 'surface'")

        exterior = _neighborCells(topology) < 0
        if selector == "all_exterior":
            return cls(entityKind=SURFACE, topology=topology, mask=exterior)
        axisSelectors = {
            "x_min": (0, np.min),
            "x_max": (0, np.max),
            "y_min": (1, np.min),
            "y_max": (1, np.max),
            "z_min": (2, np.min),
            "z_max": (2, np.max),
        }
        try:
            axis, reducer = axisSelectors[selector]
        except KeyError as exc:
            raise ValueError(f"unsupported surface selector {selector!r}") from exc
        coordinates = np.asarray(topology.faceCenters)[..., axis]
        target = reducer(coordinates[exterior])
        tolerance = max(1.0, abs(float(target))) * 1.0e-12
        selected = exterior & np.isclose(coordinates, target, rtol=0.0, atol=tolerance)
        return cls(entityKind=SURFACE, topology=topology, mask=selected)

    @classmethod
    def fromYaml(cls, filename, name, **objects):
        """Resolve one named domain from a schema-v3 YAML configuration."""
        from .configuration import objectFromYaml

        return objectFromYaml(cls, filename, name, **objects)

    @property
    def isEmpty(self):
        """Whether the region contains no selected entities."""
        return not self._shards

    @property
    def topologies(self):
        """Internal topology bindings represented by this domain."""
        return tuple(topology for topology, _mask in self._shards)

    def maskFor(self, topology):
        """Return this domain's immutable boolean selection for ``topology``."""
        for owner, mask in self._shards:
            if owner is topology:
                return mask
        empty = np.zeros(_entity_shape(topology, self.entityKind), dtype=bool)
        empty.flags.writeable = False
        return empty

    def boundary(self):
        """Return the oriented surface bounding this volume-domain union.

        A face is retained when its owning cell is selected and its neighboring
        cell is absent from the same union. Adjacency is evaluated on each
        source topology; independently meshed topologies are not geometrically
        welded by this operation.
        """
        if self.entityKind != VOLUME:
            raise TypeError("Domain.boundary requires a volume Domain")
        result = Domain(entityKind=SURFACE)
        for topology, selected in self._shards:
            neighbors = _neighborCells(topology)
            valid = neighbors >= 0
            neighbor_selected = np.zeros_like(valid, dtype=bool)
            neighbor_selected[valid] = selected[neighbors[valid]]
            exposed = selected[:, np.newaxis] & ~neighbor_selected
            result = result + Domain(
                entityKind=SURFACE,
                topology=topology,
                mask=exposed,
            )
        return result

    def __add__(self, other):
        other = Domain(other)
        self._requireSameKind(other)
        topologies = list(self.topologies)
        topologies.extend(
            topology
            for topology in other.topologies
            if all(topology is not item for item in topologies)
        )
        shards = []
        for topology in topologies:
            mask = self.maskFor(topology) | other.maskFor(topology)
            if np.any(mask):
                shards.append((topology, self._readMask(topology, mask, self.entityKind)))
        return self._fromShards(self.entityKind, shards)

    def __sub__(self, other):
        other = Domain(other)
        self._requireSameKind(other)
        shards = []
        for topology, current in self._shards:
            mask = current & ~other.maskFor(topology)
            if np.any(mask):
                shards.append((topology, self._readMask(topology, mask, self.entityKind)))
        return self._fromShards(self.entityKind, shards)

    def _requireSameKind(self, other):
        if self.entityKind != other.entityKind:
            raise TypeError("domain arithmetic requires matching entity kinds")

    @classmethod
    def _fromShards(cls, entityKind, shards):
        result = object.__new__(cls)
        object.__setattr__(result, "entityKind", entityKind)
        object.__setattr__(result, "_shards", tuple(shards))
        return result


class OpticalComponent:
    """Assign one optical material to a non-empty volume domain.

    Parameters
    ----------
    domain
        Volume :class:`Domain`. It may contain several regions or independent
        topology bindings, provided every selected entity is a volume cell.
    material
        Resolved :class:`material_library.Material` for the selected volume.
    name
        Optional component label.
    opticalRole
        Optional role metadata. Adding the component to a :class:`GainMedium`
        sets this to ``"gainElement"``.
    """

    def __init__(self, *, domain, material, name=None, opticalRole=None):
        if not isinstance(material, Material):
            raise TypeError("OpticalComponent.material must be Material")
        selected = domain if isinstance(domain, Domain) else Domain(domain)
        if selected.entityKind != VOLUME or selected.isEmpty:
            raise ValueError("OpticalComponent requires a non-empty volume Domain")
        self.material = material
        self.name = name
        self.opticalRole = opticalRole
        self._domain = selected
        self._surfaceOptics = []

    @classmethod
    def fromYaml(cls, filename, name, **objects):
        from .configuration import objectFromYaml

        return objectFromYaml(cls, filename, name, **objects)

    @property
    def domain(self):
        """Volume region occupied by this optical component."""
        return self._domain

    def getDomain(self):
        """Return :attr:`domain`; retained as a compatibility alias."""
        return self.domain

    @property
    def exteriorCells(self):
        """Volume cells touching the boundary of this component domain."""
        result = Domain(entityKind=VOLUME)
        for topology, selected in self._domain._shards:
            neighbors = _neighborCells(topology)
            local_boundary = neighbors < 0
            valid = neighbors >= 0
            neighbor_selected = np.zeros_like(valid, dtype=bool)
            neighbor_selected[valid] = selected[neighbors[valid]]
            local_boundary |= valid & ~neighbor_selected
            cells = selected & np.any(local_boundary, axis=1)
            result = result + Domain(entityKind=VOLUME, topology=topology, mask=cells)
        return result

    @property
    def exteriorTets(self):
        """Compatibility alias for :attr:`exteriorCells`."""
        return self.exteriorCells

    def assignSurfaceOptics(self, domain, optics):
        """Assign an optical boundary model to part of the component surface.

        Parameters
        ----------
        domain
            Non-empty surface domain whose owning cells belong to this
            component. Faces internal to the component are rejected.
        optics
            :class:`SurfaceOptics` applied at the selected faces.

        Returns
        -------
        OpticalComponent
            ``self`` for chained physical construction.
        """
        selected = Domain(domain)
        if selected.entityKind != SURFACE or selected.isEmpty:
            raise ValueError("surface optics require a non-empty surface Domain")
        if not isinstance(optics, SurfaceOptics):
            raise TypeError("optics must be SurfaceOptics")
        for topology, faces in selected._shards:
            component_cells = self._domain.maskFor(topology)
            if not np.any(component_cells):
                raise ValueError("surface domain belongs to a different component topology")
            neighbors = _neighborCells(topology)
            rows, local_faces = np.nonzero(faces)
            if np.any(~component_cells[rows]):
                raise ValueError("surface domain contains a face outside the component")
            neighbor = neighbors[rows, local_faces]
            internal = neighbor >= 0
            if np.any(internal & component_cells[np.maximum(neighbor, 0)]):
                raise ValueError("surface optics may only target the component boundary")
        self._surfaceOptics.append((selected, optics))
        return self

    @property
    def surfaceOptics(self):
        """Immutable sequence of ``(surfaceDomain, SurfaceOptics)`` assignments."""
        return tuple(self._surfaceOptics)


class GainMedium:
    """Select optical components participating in active gain dynamics.

    Passive transport components remain in ``Simulation.opticalComponents``
    and are not added to this container.

    Parameters
    ----------
    components
        Optical components whose materials have ``active=True`` and provide
        the run-specific ion density, fluorescence lifetime, and
        absorption/emission cross sections required by population dynamics.
    name
        Optional label for the active component group.
    """

    def __init__(self, components=(), *, name=None):
        self.name = name
        self._components = []
        for component in components:
            self.addComponent(component)

    @classmethod
    def fromYaml(cls, filename, name, **objects):
        from .configuration import objectFromYaml

        return objectFromYaml(cls, filename, name, **objects)

    def addComponent(self, component):
        """Add one active optical component and assign its gain role."""
        if not isinstance(component, OpticalComponent):
            raise TypeError("GainMedium components must be OpticalComponent")
        if component in self._components:
            raise ValueError("OpticalComponent is already part of this GainMedium")
        component.opticalRole = "gainElement"
        self._components.append(component)
        return self

    @property
    def components(self):
        """Gain components in insertion order."""
        return tuple(self._components)

    @property
    def domain(self):
        """Union of the volume domains occupied by all gain components."""
        if not self._components:
            return Domain(entityKind=VOLUME)
        return Domain(component.domain for component in self._components)

    def getDomain(self):
        """Return :attr:`domain`; retained as a compatibility alias."""
        return self.domain

__all__ = ["Domain", "GainMedium", "OpticalComponent", "SURFACE", "VOLUME"]
