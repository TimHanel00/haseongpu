# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend-neutral material, boundary, interface, and state assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .materials import MaterialInstance


def _as_tuple(value):
    if isinstance(value, (str, int, np.integer)):
        return (value,)
    return tuple(value)


@dataclass(frozen=True)
class MaterialLayout:
    domains: object

    def __post_init__(self):
        domains = _as_tuple(self.domains)
        if not domains:
            raise ValueError("MaterialLayout requires at least one volume domain")
        object.__setattr__(self, "domains", domains)


@dataclass(frozen=True)
class BoundaryLayout:
    domains: object

    def __post_init__(self):
        domains = _as_tuple(self.domains)
        if not domains:
            raise ValueError("BoundaryLayout requires at least one surface domain")
        object.__setattr__(self, "domains", domains)


@dataclass(frozen=True)
class MaterialInterfaceLayout:
    between: tuple[MaterialInstance, MaterialInstance]

    def __post_init__(self):
        between = tuple(self.between)
        if len(between) != 2 or not all(isinstance(value, MaterialInstance) for value in between):
            raise TypeError("MaterialInterfaceLayout.between must contain two MaterialInstance objects")
        if between[0] is between[1]:
            raise ValueError("a material interface requires two distinct material instances")
        object.__setattr__(self, "between", between)


class ExteriorBoundaryModel:
    """Extension role for exterior optical boundary behavior."""

    kind: ClassVar[str]


@dataclass(frozen=True)
class AbsorbingSurface(ExteriorBoundaryModel):
    """Exterior ray-stop model supported by the current backend."""

    kind: ClassVar[str] = "absorbing"


@dataclass(frozen=True)
class ConstantReflectivitySurface(ExteriorBoundaryModel):
    """Exterior constant-reflectivity model supported by the current backend."""

    reflectivity: float = 0.0
    exterior_refractive_index: float = 1.0
    kind: ClassVar[str] = "constant_reflectivity"

    def __post_init__(self):
        if not np.isfinite(self.reflectivity) or not 0.0 <= self.reflectivity <= 1.0:
            raise ValueError("surface reflectivity must be finite and within [0, 1]")
        if not np.isfinite(self.exterior_refractive_index) or self.exterior_refractive_index <= 0.0:
            raise ValueError("surface exterior_refractive_index must be finite and positive")


@dataclass(frozen=True)
class ExteriorBoundary:
    model: object
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.model, ExteriorBoundaryModel):
            raise TypeError("boundary model must implement the ExteriorBoundaryModel role")
        if not isinstance(getattr(self.model, "kind", None), str) or not self.model.kind:
            raise ValueError("exterior boundary models require a non-empty kind")


class MaterialInterfaceModel:
    """Extension role for transport behavior between unlike materials."""

    kind: ClassVar[str]


@dataclass(frozen=True)
class PerfectTransmission(MaterialInterfaceModel):
    """Future internal interface model: cross without changing direction or weight."""

    kind: ClassVar[str] = "perfect_transmission"


@dataclass(frozen=True)
class FresnelInterface(MaterialInterfaceModel):
    """Future interface model using the incident and destination material indices."""

    kind: ClassVar[str] = "fresnel"


@dataclass(frozen=True)
class MaterialInterface:
    model: object
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.model, MaterialInterfaceModel):
            raise TypeError("interface model must implement the MaterialInterfaceModel role")
        if not isinstance(getattr(self.model, "kind", None), str) or not self.model.kind:
            raise ValueError("material interface models require a non-empty kind")


@dataclass(frozen=True)
class InitialState:
    """Cell-centred initial excited-state fraction."""

    excitation_fraction: object = 0.0


@dataclass(frozen=True)
class BackendCapabilities:
    multiple_materials: bool = False
    exterior_boundary_models: frozenset[str] = frozenset({"absorbing", "constant_reflectivity"})
    internal_interface_models: frozenset[str] = frozenset()
    material_orientation: bool = False
    bulk_attenuation: bool = False
    minimum_active_materials: int = 1
    maximum_active_materials: int | None = 1

    def __post_init__(self):
        if self.minimum_active_materials < 0:
            raise ValueError("minimum_active_materials must be non-negative")
        if self.maximum_active_materials is not None and (
            self.maximum_active_materials < self.minimum_active_materials
        ):
            raise ValueError("maximum_active_materials must not be below the minimum")


CURRENT_BACKEND_CAPABILITIES = BackendCapabilities()


@dataclass(frozen=True)
class CompiledProblem:
    mesh: object
    materials: tuple[MaterialInstance, ...]
    cell_material_id: np.ndarray
    boundaries: tuple[ExteriorBoundary, ...]
    face_boundary_id: np.ndarray
    interfaces: tuple[MaterialInterface, ...]
    face_interface_id: np.ndarray
    initial_excitation_fraction: np.ndarray

    def unsupported_features(self, capabilities=CURRENT_BACKEND_CAPABILITIES):
        unsupported = []
        if len(self.materials) > 1 and not capabilities.multiple_materials:
            unsupported.append(f"multiple materials ({len(self.materials)} configured)")
        boundary_kinds = {boundary.model.kind for boundary in self.boundaries}
        missing_boundary_kinds = sorted(boundary_kinds - set(capabilities.exterior_boundary_models))
        if missing_boundary_kinds:
            unsupported.append("exterior boundary models: " + ", ".join(missing_boundary_kinds))
        kinds = {interface.model.kind for interface in self.interfaces}
        missing_kinds = sorted(kinds - set(capabilities.internal_interface_models))
        if missing_kinds:
            unsupported.append("internal interface models: " + ", ".join(missing_kinds))
        if any(material.optical_axis is not None for material in self.materials) and not capabilities.material_orientation:
            unsupported.append("material optical orientation")
        if any(material.definition.bulk_attenuation > 0.0 for material in self.materials) and not capabilities.bulk_attenuation:
            unsupported.append("per-material bulk attenuation")
        active_count = sum(material.is_active for material in self.materials)
        if active_count < capabilities.minimum_active_materials or (
            capabilities.maximum_active_materials is not None
            and active_count > capabilities.maximum_active_materials
        ):
            maximum = (
                "unbounded"
                if capabilities.maximum_active_materials is None
                else str(capabilities.maximum_active_materials)
            )
            unsupported.append(
                f"active material count {active_count} "
                f"(supported range {capabilities.minimum_active_materials}..{maximum})"
            )
        return tuple(unsupported)

    def require_backend_support(self, capabilities=CURRENT_BACKEND_CAPABILITIES):
        unsupported = self.unsupported_features(capabilities)
        if unsupported:
            raise NotImplementedError(
                "the selected HASEonGPU backend does not yet support " + "; ".join(unsupported)
            )
        return self


def _resolve_named_domain(value, names, *, kind):
    if isinstance(value, str):
        matches = [int(tag) for tag, name in names.items() if name == value]
        if not matches:
            raise KeyError(f"unknown {kind} domain name '{value}'")
        if len(matches) != 1:
            raise ValueError(f"ambiguous {kind} domain name '{value}'")
        return matches[0]
    return int(value)


def _material_mask(mesh, layout):
    if any(value == "all" for value in layout.domains):
        if len(layout.domains) != 1:
            raise ValueError("the 'all' volume selector cannot be combined with other domains")
        return np.ones(mesh.number_of_cells, dtype=bool)
    domain_ids = [_resolve_named_domain(value, mesh.volume_domain_names, kind="volume") for value in layout.domains]
    return np.isin(mesh.volume_domain_ids, domain_ids)


def _boundary_mask(mesh, layout):
    exterior = np.asarray(mesh.neighbor_cells) < 0
    if any(value == "all_exterior" for value in layout.domains):
        if len(layout.domains) != 1:
            raise ValueError("the 'all_exterior' selector cannot be combined with other domains")
        return exterior
    domain_ids = [_resolve_named_domain(value, mesh.surface_domain_names, kind="surface") for value in layout.domains]
    return exterior & np.isin(mesh.surface_domain_ids, domain_ids)


def _initial_state_array(mesh, state):
    value = state.excitation_fraction
    if isinstance(value, dict):
        result = np.full(mesh.number_of_cells, np.nan, dtype=np.float64)
        for selector, selected_value in value.items():
            layout = MaterialLayout(selector)
            mask = _material_mask(mesh, layout)
            if not np.any(mask):
                raise ValueError(f"initial-state selector {selector!r} selected no cells")
            if np.any(np.isfinite(result[mask])):
                raise ValueError(f"initial-state selector {selector!r} overlaps an earlier selector")
            result[mask] = float(selected_value)
        if np.any(~np.isfinite(result)):
            raise ValueError("domain-mapped initial state does not cover every cell")
    elif np.isscalar(value):
        result = np.full(mesh.number_of_cells, float(value), dtype=np.float64)
    else:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (mesh.number_of_cells,):
            raise ValueError(
                f"initial excitation_fraction must have shape ({mesh.number_of_cells},), got {result.shape}"
            )
        result = result.copy()
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("initial excitation_fraction must contain finite values within [0, 1]")
    return result


def compile_problem(mesh, material_registrations, boundary_registrations, interface_registrations, initial_state):
    from .mesh import UnstructuredMesh

    if not isinstance(mesh, UnstructuredMesh):
        raise TypeError("Simulation mesh must be UnstructuredMesh")
    if not isinstance(initial_state, InitialState):
        raise TypeError("Simulation initial_state must be InitialState")
    if not material_registrations:
        raise ValueError("Simulation requires at least one material registration")

    materials = []
    material_ids = {}
    cell_material_id = np.full(mesh.number_of_cells, -1, dtype=np.int32)
    for material, layout in material_registrations:
        if not isinstance(material, MaterialInstance) or not isinstance(layout, MaterialLayout):
            raise TypeError("add_material expects MaterialInstance and MaterialLayout")
        mask = _material_mask(mesh, layout)
        if not np.any(mask):
            raise ValueError(f"material layout for '{material.display_name}' selected no cells")
        if np.any(cell_material_id[mask] >= 0):
            raise ValueError(f"material layout for '{material.display_name}' overlaps an earlier layout")
        identifier = material_ids.get(material)
        if identifier is None:
            identifier = len(materials)
            material_ids[material] = identifier
            materials.append(material)
        cell_material_id[mask] = identifier
    if np.any(cell_material_id < 0):
        missing = int(np.count_nonzero(cell_material_id < 0))
        raise ValueError(f"material layouts leave {missing} cells uncovered")

    exterior = np.asarray(mesh.neighbor_cells) < 0
    face_boundary_id = np.full(mesh.neighbor_cells.shape, -1, dtype=np.int32)
    boundaries = []
    for boundary, layout in boundary_registrations:
        if not isinstance(boundary, ExteriorBoundary) or not isinstance(layout, BoundaryLayout):
            raise TypeError("add_boundary expects ExteriorBoundary and BoundaryLayout")
        mask = _boundary_mask(mesh, layout)
        if not np.any(mask):
            raise ValueError("boundary layout selected no exterior faces")
        if np.any(face_boundary_id[mask] >= 0):
            raise ValueError("boundary layout overlaps an earlier boundary layout")
        face_boundary_id[mask] = len(boundaries)
        boundaries.append(boundary)
    missing_boundary = exterior & (face_boundary_id < 0)
    if np.any(missing_boundary):
        raise ValueError(f"boundary layouts leave {int(np.count_nonzero(missing_boundary))} exterior faces uncovered")
    face_boundary_id[~exterior] = -1

    interfaces = []
    interface_by_pair = {}
    for interface, layout in interface_registrations:
        if not isinstance(interface, MaterialInterface) or not isinstance(layout, MaterialInterfaceLayout):
            raise TypeError("add_interface expects MaterialInterface and MaterialInterfaceLayout")
        try:
            pair = frozenset((material_ids[layout.between[0]], material_ids[layout.between[1]]))
        except KeyError as exc:
            raise ValueError("material interface refers to an unregistered material instance") from exc
        if pair in interface_by_pair:
            raise ValueError("duplicate material interface registration")
        interface_by_pair[pair] = len(interfaces)
        interfaces.append(interface)

    face_interface_id = np.full(mesh.neighbor_cells.shape, -1, dtype=np.int32)
    interface_face_counts = np.zeros(len(interfaces), dtype=np.int64)
    missing_pairs = set()
    for cell in range(mesh.number_of_cells):
        for local_face, neighbor in enumerate(mesh.neighbor_cells[cell]):
            neighbor = int(neighbor)
            if neighbor < 0 or neighbor < cell:
                continue
            left = int(cell_material_id[cell])
            right = int(cell_material_id[neighbor])
            if left == right:
                continue
            pair = frozenset((left, right))
            interface_id = interface_by_pair.get(pair)
            if interface_id is None:
                missing_pairs.add(tuple(sorted(pair)))
                continue
            face_interface_id[cell, local_face] = interface_id
            interface_face_counts[interface_id] += 2
            neighbor_faces = np.flatnonzero(np.asarray(mesh.neighbor_cells[neighbor]) == cell)
            if neighbor_faces.size != 1:
                raise ValueError("mesh adjacency is not reciprocal at a material interface")
            face_interface_id[neighbor, int(neighbor_faces[0])] = interface_id
    if missing_pairs:
        names = [f"{materials[a].display_name}<->{materials[b].display_name}" for a, b in sorted(missing_pairs)]
        raise ValueError("missing material interface registration for " + ", ".join(names))
    unused_interfaces = np.flatnonzero(interface_face_counts == 0)
    if unused_interfaces.size:
        names = [interfaces[int(identifier)].name or str(int(identifier)) for identifier in unused_interfaces]
        raise ValueError("material interface registrations select no adjacent faces: " + ", ".join(names))

    return CompiledProblem(
        mesh=mesh,
        materials=tuple(materials),
        cell_material_id=cell_material_id,
        boundaries=tuple(boundaries),
        face_boundary_id=face_boundary_id,
        interfaces=tuple(interfaces),
        face_interface_id=face_interface_id,
        initial_excitation_fraction=_initial_state_array(mesh, initial_state),
    )
