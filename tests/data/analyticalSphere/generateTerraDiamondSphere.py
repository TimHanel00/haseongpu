# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the painted Tet4 sphere used by the multi-domain ASE regression."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import gmsh
import numpy as np

from HASEonGPU import VolumeTopology


DEFAULT_RADIUS = 0.1
DEFAULT_MESH_SIZE_DIVISOR = 12.5
TERRA_DIAMOND_NODES = np.asarray(
    [
        [0, 5, 6, 1],
        [0, 1, 7, 2],
        [0, 2, 8, 3],
        [0, 3, 9, 4],
        [0, 4, 10, 5],
        [11, 7, 1, 6],
        [11, 8, 2, 7],
        [11, 9, 3, 8],
        [11, 10, 4, 9],
        [11, 6, 5, 10],
    ],
    dtype=np.uint32,
)


def _construct_sphere(radius: float, mesh_size_divisor: float) -> VolumeTopology:
    center = np.zeros(3, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = Path(tmpdir) / "sphere_tet4.msh"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.clear()
            gmsh.model.add("terra_diamond_sphere")
            sphere = gmsh.model.occ.addSphere(*center, radius)

            central_scale = radius / (3.0 * mesh_size_divisor)
            central_coordinates = (
                (+central_scale, +central_scale, +central_scale),
                (+central_scale, -central_scale, -central_scale),
                (-central_scale, +central_scale, -central_scale),
                (-central_scale, -central_scale, +central_scale),
            )
            points = [gmsh.model.occ.addPoint(*coordinate) for coordinate in central_coordinates]
            line01 = gmsh.model.occ.addLine(points[0], points[1])
            line02 = gmsh.model.occ.addLine(points[0], points[2])
            line03 = gmsh.model.occ.addLine(points[0], points[3])
            line12 = gmsh.model.occ.addLine(points[1], points[2])
            line13 = gmsh.model.occ.addLine(points[1], points[3])
            line23 = gmsh.model.occ.addLine(points[2], points[3])
            surfaces = [
                gmsh.model.occ.addPlaneSurface(
                    [gmsh.model.occ.addCurveLoop([line02, -line12, -line01])]
                ),
                gmsh.model.occ.addPlaneSurface(
                    [gmsh.model.occ.addCurveLoop([line01, line13, -line03])]
                ),
                gmsh.model.occ.addPlaneSurface(
                    [gmsh.model.occ.addCurveLoop([line03, -line23, -line02])]
                ),
                gmsh.model.occ.addPlaneSurface(
                    [gmsh.model.occ.addCurveLoop([line12, line23, -line13])]
                ),
            ]
            central_tetrahedron = gmsh.model.occ.addVolume(
                [gmsh.model.occ.addSurfaceLoop(surfaces)]
            )
            fragments, _ = gmsh.model.occ.fragment(
                [(3, sphere)],
                [(3, central_tetrahedron)],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            volume_tags = [tag for dimension, tag in fragments if dimension == 3]
            gmsh.model.addPhysicalGroup(3, volume_tags, 1)
            mesh_size = max(radius / mesh_size_divisor, 1.0e-5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.model.mesh.generate(3)
            gmsh.write(str(mesh_path))
        finally:
            gmsh.finalize()
        return VolumeTopology.fromFile(mesh_path)


def _icosahedron_nodes() -> np.ndarray:
    fifth_pi = math.pi / 5.0
    angle = 2.0 * math.acos(1.0 / (2.0 * math.sin(fifth_pi)))
    nodes = [[0.0, 0.0, 1.0]]
    nodes.extend(
        [
            math.sin(angle) * math.cos(2.0 * (index - 0.5) * fifth_pi),
            math.sin(angle) * math.sin(2.0 * (index - 0.5) * fifth_pi),
            math.cos(angle),
        ]
        for index in range(1, 6)
    )
    nodes.extend(
        [
            math.sin(angle) * math.cos(2.0 * (index - 1.0) * fifth_pi),
            math.sin(angle) * math.sin(2.0 * (index - 1.0) * fifth_pi),
            -math.cos(angle),
        ]
        for index in range(1, 6)
    )
    nodes.append([0.0, 0.0, -1.0])
    return np.asarray(nodes, dtype=np.float64)


def _diamond_face_normals() -> tuple[np.ndarray, np.ndarray]:
    nodes = _icosahedron_nodes()
    normals = []
    diamond_ids = []
    for diamond_id, corners in enumerate(TERRA_DIAMOND_NODES):
        pole, first, opposite, second = corners
        left, right = (first, second) if diamond_id < 5 else (second, first)
        for face in ((pole, left, right), (left, opposite, right)):
            vertices = nodes[np.asarray(face)]
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if np.dot(normal, vertices.mean(axis=0)) < 0.0:
                normal *= -1.0
            normals.append(normal / np.linalg.norm(normal))
            diamond_ids.append(diamond_id)
    return np.asarray(normals), np.asarray(diamond_ids, dtype=np.uint32)


def _classify_diamonds(cell_centers: np.ndarray) -> np.ndarray:
    normals, face_diamond_ids = _diamond_face_normals()
    centers = np.asarray(cell_centers, dtype=np.float64)
    radii = np.linalg.norm(centers, axis=1)
    directions = np.zeros_like(centers)
    noncentral = radii > 64.0 * np.finfo(np.float64).eps
    directions[noncentral] = centers[noncentral] / radii[noncentral, np.newaxis]
    face_ids = np.argmax(directions @ normals.T, axis=1)
    diamond_ids = face_diamond_ids[face_ids]
    diamond_ids[~noncentral] = 0
    return diamond_ids


def _write_vtk(path: Path, topology: VolumeTopology, diamond_ids: np.ndarray) -> None:
    points = np.asarray(topology.points, dtype=np.float64)
    cells = np.asarray(topology.cellPointIndices, dtype=np.uint32)
    with path.open("w", encoding="utf-8") as output:
        output.write("# vtk DataFile Version 2.0\n")
        output.write("HASEonGPU analytical sphere with TERRA-NG diamond domains\n")
        output.write("ASCII\n")
        output.write("DATASET UNSTRUCTURED_GRID\n")
        output.write(f"POINTS {points.shape[0]} double\n")
        for point in points:
            output.write(" ".join(f"{coordinate:.17g}" for coordinate in point) + "\n")
        output.write(f"CELLS {cells.shape[0]} {5 * cells.shape[0]}\n")
        for cell in cells:
            output.write("4 " + " ".join(str(int(vertex)) for vertex in cell) + "\n")
        output.write(f"CELL_TYPES {cells.shape[0]}\n")
        output.write("10\n" * cells.shape[0])
        output.write(f"CELL_DATA {cells.shape[0]}\n")
        for name in ("diamondId", "cellDomains"):
            output.write(f"SCALARS {name} unsigned_int 1\n")
            output.write("LOOKUP_TABLE default\n")
            output.write("\n".join(str(int(value)) for value in diamond_ids) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("terraDiamondSphere.vtk"),
    )
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument(
        "--mesh-size-divisor",
        type=float,
        default=DEFAULT_MESH_SIZE_DIVISOR,
    )
    arguments = parser.parse_args()
    topology = _construct_sphere(arguments.radius, arguments.mesh_size_divisor)
    diamond_ids = _classify_diamonds(topology.cellCenters)
    if set(np.unique(diamond_ids)) != set(range(10)):
        raise RuntimeError("generated sphere does not contain all ten TERRA-NG diamonds")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    _write_vtk(arguments.output, topology, diamond_ids)
    counts = np.bincount(diamond_ids, minlength=10)
    print(f"wrote {arguments.output}: {topology.numberOfCells} cells, diamond counts={counts.tolist()}")


if __name__ == "__main__":
    main()
