# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tet4 mesh topology for the public Python API."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .geometry.volume import VolumeTopology
from hase_units import Unit


@dataclass(frozen=True, eq=False)
class MeshSelection:
    """Typed selection of cells or exterior triangular faces from one mesh.

    Selections are created by :meth:`UnstructuredMesh.volume`,
    :meth:`UnstructuredMesh.surface`, :attr:`UnstructuredMesh.exteriorFaces`,
    or :attr:`UnstructuredMesh.boundaryCells`; users normally do not construct
    them directly. A selection retains its originating mesh so registrations
    cannot accidentally combine domains from different topologies.

    Parameters
    ----------
    mesh
        The owning :class:`UnstructuredMesh`.
    kind
        ``"volume"`` for Tet4 cells or ``"surface"`` for exterior faces.
    names
        Human-readable domain names represented by the selection.
    _mask
        Boolean cell mask with shape ``(numberOfCells,)`` or face mask with
        shape ``(numberOfCells, 4)``.
    """

    mesh: object
    """Mesh that owns the selected cells or faces and their index layout."""
    kind: str
    """Entity category: ``"volume"`` for cells or ``"surface"`` for faces."""
    names: tuple[str, ...]
    """Human-readable domain labels represented by this union selection."""
    _mask: object

    def __post_init__(self):
        if self.kind not in {"volume", "surface"}:
            raise ValueError("domain selection kind must be 'volume' or 'surface'")
        mask = np.asarray(self._mask, dtype=bool).copy()
        expected = (
            (self.mesh.numberOfCells,)
            if self.kind == "volume"
            else np.asarray(self.mesh.neighborCells).shape
        )
        if mask.shape != expected:
            raise ValueError(f"{self.kind} selection mask must have shape {expected}, got {mask.shape}")
        if not np.any(mask):
            raise ValueError("mesh selection must not be empty")
        mask.flags.writeable = False
        object.__setattr__(self, "_mask", mask)

    def mask(self):
        """Return the immutable boolean mask in the mesh's native layout."""
        return self._mask

    @property
    def indices(self):
        """Return selected cell ids or ``(cell_id, local_face_id)`` pairs."""
        indices = np.argwhere(self._mask)
        return indices[:, 0] if self.kind == "volume" else indices


class UnstructuredMesh(VolumeTopology):
    """Unstructured Tet4 topology with named volume and surface domains.

    The mesh contains topology, geometry scale, and domain identity only.
    Material, boundary, interface, and pump behavior is registered separately
    on a :class:`Simulation` by using typed :class:`MeshSelection` objects.

    Point coordinates are stored as plain numbers; ``coordinateUnit`` gives
    them physical length. All public array views are read-only.

    Notes
    -----
    Every cell is a four-node tetrahedron. Surface-domain arrays use shape
    ``(numberOfCells, 4)`` in the topology's local-face order. A value of
    zero means that the face has no named surface domain; only exterior faces
    can be returned by :meth:`surface`.

    """

    def __post_init__(self):
        coordinateUnit = self.metadata.get("coordinate_unit")
        if not isinstance(coordinateUnit, Unit):
            raise TypeError("mesh metadata must declare coordinateUnit as a Unit")
        points = np.asarray(self.points, dtype=np.float64)
        cells = np.asarray(self.cellPointIndices)
        if np.any(~np.isfinite(points)):
            raise ValueError("mesh points must be finite")
        if cells.ndim == 2 and cells.shape[1] == 4:
            if not np.issubdtype(cells.dtype, np.integer):
                raise TypeError("Tet4 connectivity must contain integer point indices")
            if np.any(cells < 0) or np.any(cells >= points.shape[0]):
                raise ValueError("Tet4 connectivity contains an out-of-range point index")
            if any(np.unique(cell).size != 4 for cell in cells):
                raise ValueError("each Tet4 cell must reference four distinct points")
        self.points = points.copy()
        self.cellPointIndices = cells.copy()
        for name in ("cellTypes", "cellDomains", "faceBoundaries"):
            values = getattr(self, name, None)
            if values is not None:
                setattr(self, name, np.asarray(values).copy())
        super().__post_init__()
        coordinates = np.asarray(self.points)[np.asarray(self.cellPointIndices)]
        signed_six_volume = np.einsum(
            "ij,ij->i",
            np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
            coordinates[:, 3] - coordinates[:, 0],
        )
        if np.any(~np.isfinite(signed_six_volume)) or np.any(signed_six_volume == 0.0):
            raise ValueError("Tet4 cells must have finite, non-zero volume")
        face_keys = np.sort(np.asarray(self.facePointIndices).reshape(-1, 3), axis=1)
        _keys, counts = np.unique(face_keys, axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Tet4 mesh contains a non-manifold face shared by more than two cells")
        # The legacy topology overloads -1 as a ray-stop behavior. The public
        # mesh uses 0 solely to mean "no named surface domain".
        self.faceBoundaries = np.where(np.asarray(self.faceBoundaries) > 0, self.faceBoundaries, 0).astype(
            np.int32, copy=False
        )
        for name in (
            "points",
            "cellPointIndices",
            "cellTypes",
            "cellDomains",
            "faceBoundaries",
            "facePointIndices",
            "neighborCells",
            "neighborLocalFaces",
            "faceCenters",
            "faceNormals",
            "faceAreas",
            "cellCenters",
            "cellVolumes",
            "samplePoints",
        ):
            values = getattr(self, name, None)
            if isinstance(values, np.ndarray):
                values.flags.writeable = False

    @classmethod
    def _from_volume_topology(cls, topology):
        mesh = cls(
            points=np.asarray(topology.points).copy(),
            cellPointIndices=np.asarray(topology.cellPointIndices).copy(),
            cellTypes=np.asarray(topology.cellTypes).copy(),
            cellDomains=np.asarray(topology.cellDomains).copy(),
            faceBoundaries=np.where(np.asarray(topology.faceBoundaries) > 0, topology.faceBoundaries, 0),
            metadata=dict(topology.metadata),
        )
        mesh.samplePoints = np.asarray(topology.samplePoints, dtype=np.float64).copy()
        mesh.samplePoints.flags.writeable = False
        return mesh

    @classmethod
    def fromTetrahedra(
        cls,
        points,
        cellConnectivity,
        *,
        volumeDomains=None,
        surfaceDomains=None,
        volumeDomainNames=None,
        surfaceDomainNames=None,
        metadata=None,
        coordinateUnit=None,
    ):
        """Construct a Tet4 mesh from points, connectivity, and domain tags.

        Parameters
        ----------
        points
            Finite coordinates with shape ``(numberOfPoints, 3)``. Their
            numeric values are interpreted in ``coordinateUnit``.
        cellConnectivity
            Integer point ids with shape ``(numberOfCells, 4)``.
        volumeDomains
            Optional integer physical-domain tag per cell. These tags identify
            regions but do not assign material behavior by themselves.
        surfaceDomains
            Optional integer tags with shape ``(numberOfCells, 4)`` in local
            face order. A zero denotes no named surface group.
        volumeDomainNames, surfaceDomainNames
            Optional mappings from integer tags to physical-group names.
        metadata
            Additional topology metadata copied into the mesh.
        coordinateUnit
            Required physical length unit for the numeric point coordinates;
            for example, ``units.mm`` makes a coordinate difference of ``1``
            equal to one millimetre.

        Returns
        -------
        UnstructuredMesh
            Validated, immutable Tet4 topology.
        """
        if not isinstance(coordinateUnit, Unit):
            raise TypeError("fromTetrahedra requires coordinateUnit=units.<length unit>")
        merged_metadata = dict(metadata or {})
        merged_metadata["coordinate_unit"] = coordinateUnit
        if volumeDomainNames:
            merged_metadata["cellDomainNames"] = {int(tag): str(name) for tag, name in volumeDomainNames.items()}
        if surfaceDomainNames:
            merged_metadata["surfaceDomainNames"] = {
                int(tag): str(name) for tag, name in surfaceDomainNames.items()
            }
        return cls(
            points=np.asarray(points),
            cellPointIndices=np.asarray(cellConnectivity),
            cellDomains=volumeDomains,
            faceBoundaries=surfaceDomains,
            metadata=merged_metadata,
        )

    @classmethod
    def fromFile(cls, filename, *, format=None, coordinateUnit=None, **kwargs):
        """Load a Tet4 mesh from Gmsh, VTK, or a closed STL surface.

        Parameters
        ----------
        filename
            Input path. The suffix selects ``msh``, ``vtk``, or ``stl`` when
            ``format`` is omitted.
        format
            Optional explicit format name.
        coordinateUnit
            Required keyword-only physical length unit for file coordinates.
        **kwargs
            Format-specific options forwarded to the Gmsh/STL loader, such as
            ``meshSize`` for tetrahedralizing a closed STL surface.

        Returns
        -------
        UnstructuredMesh
            Loaded Tet4 topology with Gmsh physical names preserved.
        """
        path = Path(filename)
        if not isinstance(coordinateUnit, Unit):
            raise TypeError("fromFile requires coordinateUnit=units.<length unit>")
        mesh_format = (format or path.suffix.lstrip(".")).lower()
        if mesh_format in {"msh", "gmsh"}:
            topology = VolumeTopology.fromGmsh(path, **kwargs)
        elif mesh_format == "vtk":
            topology = VolumeTopology.fromVtk(path)
        elif mesh_format in {"stl", "ascii-stl", "binary-stl", "dae/stl", "dea/stl"}:
            topology = VolumeTopology.fromStl(path, **kwargs)
        else:
            raise NotImplementedError(
                f"unstructured mesh format '{mesh_format}' is not supported; supported formats: gmsh, vtk, stl"
            )
        topology.metadata["coordinate_unit"] = coordinateUnit
        return cls._from_volume_topology(topology)

    @property
    def coordinateUnit(self):
        """Physical length represented by one numeric coordinate unit."""
        return self.metadata["coordinate_unit"]

    def _select_domains(self, kind, values):
        if not values:
            raise ValueError(f"{kind} requires at least one domain name or id")
        names = self.volumeDomainNames if kind == "volume" else self.surfaceDomainNames
        ids = []
        shown_names = []
        for value in values:
            if isinstance(value, str):
                matches = [int(tag) for tag, name in names.items() if name == value]
                if not matches:
                    raise KeyError(f"unknown {kind} domain '{value}'")
                if len(matches) != 1:
                    raise ValueError(f"ambiguous {kind} domain '{value}'")
                identifier = matches[0]
                shown_names.append(value)
            else:
                identifier = int(value)
                shown_names.append(names.get(identifier, str(identifier)))
            ids.append(identifier)
        mask = (
            np.isin(self.volumeDomainIds, ids)
            if kind == "volume"
            else (np.asarray(self.neighborCells) < 0) & np.isin(self.surfaceDomainIds, ids)
        )
        try:
            return MeshSelection(self, kind, tuple(shown_names), mask)
        except ValueError as exc:
            raise ValueError(f"{kind} selection {tuple(shown_names)!r} contains no mesh entities") from exc

    def volume(self, *domains):
        """Select cells belonging to one or more named or tagged domains.

        Parameters
        ----------
        *domains
            One or more physical-group names or integer volume tags. The
            resulting selection is their union.

        Returns
        -------
        MeshSelection
            Volume-kind selection owned by this mesh.

        Examples
        --------
        ``mesh.volume("gain")`` selects one named physical volume, while
        ``mesh.volume(10, 20)`` selects the union of two numeric tags.
        """
        return self._select_domains("volume", domains)

    def surface(self, *domains):
        """Select exterior faces in one or more named or tagged domains.

        Parameters
        ----------
        *domains
            One or more physical-group names or integer surface tags.

        Returns
        -------
        MeshSelection
            Surface-kind union selection owned by this mesh.

        Internal faces are excluded even if the underlying file assigned a
        matching tag. The returned mask has the mesh's ``(cell, local_face)``
        layout.
        """
        return self._select_domains("surface", domains)

    def surfaces(self, *domains):
        """Alias for :meth:`surface` for readability with several domains.

        Parameters
        ----------
        *domains
            Physical-group names or integer surface tags, with the same union
            semantics as :meth:`surface`.
        """
        return self.surface(*domains)

    @property
    def exteriorFaces(self):
        """Selection containing every exterior face, including untagged ones."""
        return MeshSelection(
            self,
            "surface",
            ("exterior",),
            np.asarray(self.neighborCells) < 0,
        )

    @property
    def boundaryCells(self):
        """Volume selection of cells touching at least one exterior face."""
        return MeshSelection(
            self,
            "volume",
            ("boundaryCells",),
            np.any(np.asarray(self.neighborCells) < 0, axis=1),
        )

    @property
    def cellConnectivity(self):
        """Read-only ``(numberOfCells, 4)`` array of vertex ids."""
        return self.cellPointIndices

    @property
    def volumeDomainIds(self):
        """Read-only integer volume-domain tag for every cell."""
        return self.cellDomains

    @property
    def surfaceDomainIds(self):
        """Read-only ``(cell, local_face)`` surface-domain tags."""
        return self.faceBoundaries

    @property
    def volumeDomainNames(self):
        """Mapping from numeric volume tags to physical-group names."""
        return self.cellDomainNames

    @property
    def surfaceDomainNames(self):
        """Mapping from numeric surface tags to physical-group names."""
        return super().surfaceDomainNames
