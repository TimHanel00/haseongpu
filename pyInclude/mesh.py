# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tet4 mesh topology for the public Python API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry.volume import VolumeTopology


class UnstructuredMesh(VolumeTopology):
    """Unstructured Tet4 topology with named volume and surface domains.

    The mesh contains topology and domain identity only. Material, boundary,
    and interface behavior is attached later through ``Simulation`` layouts.
    """

    def __post_init__(self):
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
        return cls(
            points=np.asarray(topology.points).copy(),
            cellPointIndices=np.asarray(topology.cellPointIndices).copy(),
            cellTypes=np.asarray(topology.cellTypes).copy(),
            cellDomains=np.asarray(topology.cellDomains).copy(),
            faceBoundaries=np.where(np.asarray(topology.faceBoundaries) > 0, topology.faceBoundaries, 0),
            metadata=dict(topology.metadata),
        )

    @classmethod
    def from_tetrahedra(
        cls,
        points,
        cell_connectivity,
        *,
        volume_domains=None,
        surface_domains=None,
        volume_domain_names=None,
        surface_domain_names=None,
        metadata=None,
    ):
        merged_metadata = dict(metadata or {})
        if volume_domain_names:
            merged_metadata["cellDomainNames"] = {int(tag): str(name) for tag, name in volume_domain_names.items()}
        if surface_domain_names:
            merged_metadata["surfaceDomainNames"] = {
                int(tag): str(name) for tag, name in surface_domain_names.items()
            }
        return cls(
            points=np.asarray(points),
            cellPointIndices=np.asarray(cell_connectivity),
            cellDomains=volume_domains,
            faceBoundaries=surface_domains,
            metadata=merged_metadata,
        )

    @classmethod
    def from_file(cls, filename, *, format=None, **kwargs):
        path = Path(filename)
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
        return cls._from_volume_topology(topology)

    @property
    def number_of_points(self):
        return self.numberOfPoints

    @property
    def number_of_cells(self):
        return self.numberOfCells

    @property
    def number_of_faces_per_cell(self):
        return self.numberOfFacesPerCell

    @property
    def cell_connectivity(self):
        return self.cellPointIndices

    @property
    def volume_domain_ids(self):
        return self.cellDomains

    @property
    def surface_domain_ids(self):
        return self.faceBoundaries

    @property
    def neighbor_cells(self):
        return self.neighborCells

    @property
    def neighbor_local_faces(self):
        return self.neighborLocalFaces

    @property
    def volume_domain_names(self):
        return self.cellDomainNames

    @property
    def surface_domain_names(self):
        return self.surfaceDomainNames
