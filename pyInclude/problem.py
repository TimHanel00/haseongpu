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

from material_library import MaterialCondition
from .mesh import MeshSelection
from hase_units import DIMENSIONLESS, Quantity, requireQuantity, units


class ExteriorBoundaryModel:
    """Base role for optical behavior on an exterior mesh face.

    Custom descriptors can derive from this class, but the current native
    adapter accepts only the kinds listed by :class:`BackendCapabilities`.
    """

    kind: ClassVar[str]
    """Stable boundary-model discriminator used by backend capability checks."""


@dataclass(frozen=True)
class AbsorbingSurface(ExteriorBoundaryModel):
    """Terminate incident transport at an exterior boundary.

    No reflected ray is launched and any remaining ray weight leaves the
    simulated optical system.
    """

    kind: ClassVar[str] = "absorbing"
    """Boundary-model discriminator indicating complete ray termination."""


@dataclass(frozen=True)
class ConstantReflectivitySurface(ExteriorBoundaryModel):
    """Reflect a constant fraction of incident weight at an exterior face.

    Parameters
    ----------
    reflectivity
        Dimensionless reflected fraction in ``[0, 1]``. It is independent of
        incidence angle and polarization.
    exteriorRefractiveIndex
        Positive dimensionless index outside the mesh. Together with the
        interior material index it determines total internal reflection.
    """

    reflectivity: float = 0.0
    """Angle-independent fraction of incident ray weight that is reflected."""
    exteriorRefractiveIndex: float = 1.0
    """Dimensionless optical index immediately outside the mesh boundary."""
    kind: ClassVar[str] = "constant_reflectivity"
    """Boundary-model discriminator for angle-independent reflectivity."""

    def __post_init__(self):
        if not np.isfinite(self.reflectivity) or not 0.0 <= self.reflectivity <= 1.0:
            raise ValueError("surface reflectivity must be finite and within [0, 1]")
        if not np.isfinite(self.exteriorRefractiveIndex) or self.exteriorRefractiveIndex <= 0.0:
            raise ValueError("surface exteriorRefractiveIndex must be finite and positive")


class MaterialInterfaceModel:
    """Base role for transport between adjacent material conditions."""

    kind: ClassVar[str]
    """Stable interface-model discriminator used by capability checks."""


@dataclass(frozen=True)
class PerfectTransmission(MaterialInterfaceModel):
    """Cross an internal interface without changing direction or weight.

    This model resolves into frontend tables but is not yet executable by the
    current native adapter.
    """

    kind: ClassVar[str] = "perfect_transmission"
    """Interface discriminator for unchanged direction and ray weight."""


@dataclass(frozen=True)
class FresnelInterface(MaterialInterfaceModel):
    """Use adjacent material indices for Fresnel interface transport.

    This descriptor expresses frontend intent; current native transport does
    not yet execute cross-material interfaces.
    """

    kind: ClassVar[str] = "fresnel"
    """Interface discriminator for refractive-index-based Fresnel transport."""


@dataclass(frozen=True)
class InitialState:
    """Initial upper-state population fraction on Tet4 cells.

    Parameters
    ----------
    excitationFraction
        Dimensionless :class:`Quantity` in ``[0, 1]``. Supply one scalar, an
        array with shape ``(mesh.numberOfCells,)``, or a mapping from complete,
        non-overlapping volume :class:`MeshSelection` objects to scalar
        dimensionless quantities.

    Examples
    --------
    ``InitialState(0.1 * units.one)`` initializes every cell uniformly.
    """

    excitationFraction: object = Quantity(0.0, units.one)
    """Initial dimensionless upper-state population fraction on every cell."""


@dataclass(frozen=True)
class BackendCapabilities:
    """Feature subset executable by a backend adapter.

    Parameters
    ----------
    multipleMaterials
        Whether more than one resolved material condition can be transported.
    exteriorBoundaryModels
        Supported :attr:`ExteriorBoundaryModel.kind` strings.
    internalInterfaceModels
        Supported :attr:`MaterialInterfaceModel.kind` strings.
    materialOrientation
        Whether non-null material optical axes are executable.
    bulkAttenuation
        Whether per-material passive attenuation is executable.
    minimumActiveMaterials, maximumActiveMaterials
        Inclusive supported range of materials with positive active-ion
        density. ``maximumActiveMaterials=None`` means unbounded.
    """
    multipleMaterials: bool = False
    """Whether the adapter can transport more than one material condition."""
    exteriorBoundaryModels: frozenset[str] = frozenset({"absorbing", "constant_reflectivity"})
    """Exterior boundary-model kind strings accepted by the adapter."""
    internalInterfaceModels: frozenset[str] = frozenset()
    """Internal material-interface kind strings accepted by the adapter."""
    materialOrientation: bool = False
    """Whether non-null material optical axes are executable."""
    bulkAttenuation: bool = False
    """Whether passive per-material bulk attenuation is executable."""
    minimumActiveMaterials: int = 1
    """Minimum number of positive-active-ion-density materials required."""
    maximumActiveMaterials: int | None = 1
    """Maximum active-material count, or ``None`` when unbounded."""

    def __post_init__(self):
        if self.minimumActiveMaterials < 0:
            raise ValueError("minimumActiveMaterials must be non-negative")
        if self.maximumActiveMaterials is not None and (
            self.maximumActiveMaterials < self.minimumActiveMaterials
        ):
            raise ValueError("maximumActiveMaterials must not be below the minimum")


#: Feature declaration enforced by the currently shipped native adapter.
currentBackendCapabilities = BackendCapabilities()


@dataclass(frozen=True)
class ResolvedProblem:
    """Validated backend-neutral tables produced by :meth:`Simulation.resolveProblem`.

    Parameters
    ----------
    mesh
        Source :class:`UnstructuredMesh`.
    materials
        Dense tuple of distinct resolved material conditions.
    cellMaterialId
        Integer material-table id for every Tet4 cell.
    boundaries
        Dense tuple of distinct exterior boundary registrations.
    faceBoundaryId
        Boundary-table ids in ``(cell, local_face)`` layout.
    interfaces
        Dense tuple of internal interface registrations.
    faceInterfaceId
        Interface-table ids in ``(cell, local_face)`` layout.
    initialExcitationFraction
        Dimensionless floating-point initial fraction for every cell.

    Notes
    -----
    Compilation validates physical coverage without launching native code.
    Call :meth:`requireBackendSupport` separately to enforce one adapter's
    current execution subset.
    """
    mesh: object
    """Source mesh defining all cell and local-face indices."""
    materials: tuple[MaterialCondition, ...]
    """Distinct resolved material conditions in dense table order."""
    cellMaterialId: np.ndarray
    """Material-table id for every Tet4 cell."""
    boundaries: tuple[ExteriorBoundaryModel, ...]
    """Distinct exterior boundary descriptors in dense table order."""
    faceBoundaryId: np.ndarray
    """Boundary-table ids in ``(cell, localFace)`` layout; ``-1`` internally."""
    interfaces: tuple[MaterialInterfaceModel, ...]
    """Distinct internal interface descriptors in dense table order."""
    faceInterfaceId: np.ndarray
    """Interface-table ids in ``(cell, localFace)`` layout; ``-1`` otherwise."""
    initialExcitationFraction: np.ndarray
    """Dimensionless initial upper-state population fraction for every cell."""

    def unsupportedFeatures(self, capabilities=currentBackendCapabilities):
        """Return descriptions of resolved features absent from ``capabilities``.

        Parameters
        ----------
        capabilities
            Backend feature declaration to compare against.

        Returns
        -------
        tuple[str, ...]
            Empty when all resolved features are executable.
        """
        unsupported = []
        if len(self.materials) > 1 and not capabilities.multipleMaterials:
            unsupported.append(f"multiple materials ({len(self.materials)} configured)")
        boundary_kinds = {boundary.kind for boundary in self.boundaries}
        missing_boundary_kinds = sorted(boundary_kinds - set(capabilities.exteriorBoundaryModels))
        if missing_boundary_kinds:
            unsupported.append("exterior boundary models: " + ", ".join(missing_boundary_kinds))
        kinds = {interface.kind for interface in self.interfaces}
        missing_kinds = sorted(kinds - set(capabilities.internalInterfaceModels))
        if missing_kinds:
            unsupported.append("internal interface models: " + ", ".join(missing_kinds))
        if any(material.opticalAxis is not None for material in self.materials) and not capabilities.materialOrientation:
            unsupported.append("material optical orientation")
        if any(float(material.bulkAttenuation.siValue) > 0.0 for material in self.materials) and not capabilities.bulkAttenuation:
            unsupported.append("per-material bulk attenuation")
        active_count = sum(material.isActive for material in self.materials)
        if active_count < capabilities.minimumActiveMaterials or (
            capabilities.maximumActiveMaterials is not None
            and active_count > capabilities.maximumActiveMaterials
        ):
            maximum = (
                "unbounded"
                if capabilities.maximumActiveMaterials is None
                else str(capabilities.maximumActiveMaterials)
            )
            unsupported.append(
                f"active material count {active_count} "
                f"(supported range {capabilities.minimumActiveMaterials}..{maximum})"
            )
        return tuple(unsupported)

    def requireBackendSupport(self, capabilities=currentBackendCapabilities):
        """Return this problem or raise for unsupported resolved features.

        Parameters
        ----------
        capabilities
            Backend feature declaration to enforce.

        Raises
        ------
        NotImplementedError
            If :meth:`unsupportedFeatures` is non-empty.
        """
        unsupported = self.unsupportedFeatures(capabilities)
        if unsupported:
            raise NotImplementedError(
                "the selected HASEonGPU backend does not yet support " + "; ".join(unsupported)
            )
        return self


def _selection_mask(mesh, selection, kind):
    if not isinstance(selection, MeshSelection):
        raise TypeError(f"domains must be a mesh.{kind}(...) selection")
    if selection.mesh is not mesh:
        raise ValueError("domain selection belongs to a different mesh")
    if selection.kind != kind:
        raise TypeError(f"expected a {kind} domain selection, got {selection.kind}")
    return selection.mask()


def _initial_state_array(mesh, state):
    value = state.excitationFraction
    if isinstance(value, dict):
        result = np.full(mesh.numberOfCells, np.nan, dtype=np.float64)
        for selector, selected_value in value.items():
            mask = _selection_mask(mesh, selector, "volume")
            if not np.any(mask):
                raise ValueError(f"initial-state selector {selector!r} selected no cells")
            if np.any(np.isfinite(result[mask])):
                raise ValueError(f"initial-state selector {selector!r} overlaps an earlier selector")
            selected_value = requireQuantity(
                selected_value,
                DIMENSIONLESS,
                "initial excitationFraction",
            )
            result[mask] = float(selected_value.toValue(units.one))
        if np.any(~np.isfinite(result)):
            raise ValueError("domain-mapped initial state does not cover every cell")
    else:
        value = requireQuantity(value, DIMENSIONLESS, "initial excitationFraction")
        raw = np.asarray(value.toValue(units.one), dtype=np.float64)
        if raw.ndim == 0:
            result = np.full(mesh.numberOfCells, float(raw), dtype=np.float64)
        else:
            result = raw
        if result.shape != (mesh.numberOfCells,):
            raise ValueError(
                f"initial excitationFraction must have shape ({mesh.numberOfCells},), got {result.shape}"
            )
        result = result.copy()
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("initial excitationFraction must contain finite values within [0, 1]")
    return result


def resolve_problem(mesh, material_registrations, boundary_registrations, interface_registrations, initial_state):
    from .mesh import UnstructuredMesh

    if not isinstance(mesh, UnstructuredMesh):
        raise TypeError("Simulation mesh must be UnstructuredMesh")
    if not isinstance(initial_state, InitialState):
        raise TypeError("Simulation initial_state must be InitialState")
    if not material_registrations:
        raise ValueError("Simulation requires at least one material registration")

    materials = []
    material_ids = {}
    cellMaterialId = np.full(mesh.numberOfCells, -1, dtype=np.int32)
    for material, domains in material_registrations:
        if not isinstance(material, MaterialCondition):
            raise TypeError("addMaterial expects MaterialCondition")
        mask = _selection_mask(mesh, domains, "volume")
        if not np.any(mask):
            raise ValueError(f"material layout for '{material.displayName}' selected no cells")
        if np.any(cellMaterialId[mask] >= 0):
            raise ValueError(f"material layout for '{material.displayName}' overlaps an earlier layout")
        identifier = material_ids.get(material)
        if identifier is None:
            identifier = len(materials)
            material_ids[material] = identifier
            materials.append(material)
        cellMaterialId[mask] = identifier
    if np.any(cellMaterialId < 0):
        missing = int(np.count_nonzero(cellMaterialId < 0))
        raise ValueError(f"material layouts leave {missing} cells uncovered")

    exterior = np.asarray(mesh.neighborCells) < 0
    faceBoundaryId = np.full(mesh.neighborCells.shape, -1, dtype=np.int32)
    boundaries = []
    for boundary, domains in boundary_registrations:
        if not isinstance(boundary, ExteriorBoundaryModel):
            raise TypeError("addBoundary expects an ExteriorBoundaryModel")
        mask = _selection_mask(mesh, domains, "surface")
        if not np.any(mask):
            raise ValueError("boundary layout selected no exterior faces")
        if np.any(faceBoundaryId[mask] >= 0):
            raise ValueError("boundary layout overlaps an earlier boundary layout")
        faceBoundaryId[mask] = len(boundaries)
        boundaries.append(boundary)
    missing_boundary = exterior & (faceBoundaryId < 0)
    if np.any(missing_boundary):
        raise ValueError(f"boundary layouts leave {int(np.count_nonzero(missing_boundary))} exterior faces uncovered")
    faceBoundaryId[~exterior] = -1

    interfaces = []
    interface_by_pair = {}
    for interface, between in interface_registrations:
        if not isinstance(interface, MaterialInterfaceModel):
            raise TypeError("addInterface expects a MaterialInterfaceModel")
        between = tuple(between)
        if len(between) != 2 or not all(isinstance(value, MaterialCondition) for value in between):
            raise TypeError("between must contain two MaterialCondition objects")
        if between[0] is between[1]:
            raise ValueError("a material interface requires two distinct material conditions")
        try:
            pair = frozenset((material_ids[between[0]], material_ids[between[1]]))
        except KeyError as exc:
            raise ValueError("material interface refers to an unregistered material condition") from exc
        if pair in interface_by_pair:
            raise ValueError("duplicate material interface registration")
        interface_by_pair[pair] = len(interfaces)
        interfaces.append(interface)

    faceInterfaceId = np.full(mesh.neighborCells.shape, -1, dtype=np.int32)
    interface_face_counts = np.zeros(len(interfaces), dtype=np.int64)
    missing_pairs = set()
    for cell in range(mesh.numberOfCells):
        for local_face, neighbor in enumerate(mesh.neighborCells[cell]):
            neighbor = int(neighbor)
            if neighbor < 0 or neighbor < cell:
                continue
            left = int(cellMaterialId[cell])
            right = int(cellMaterialId[neighbor])
            if left == right:
                continue
            pair = frozenset((left, right))
            interface_id = interface_by_pair.get(pair)
            if interface_id is None:
                missing_pairs.add(tuple(sorted(pair)))
                continue
            faceInterfaceId[cell, local_face] = interface_id
            interface_face_counts[interface_id] += 2
            neighbor_faces = np.flatnonzero(np.asarray(mesh.neighborCells[neighbor]) == cell)
            if neighbor_faces.size != 1:
                raise ValueError("mesh adjacency is not reciprocal at a material interface")
            faceInterfaceId[neighbor, int(neighbor_faces[0])] = interface_id
    if missing_pairs:
        names = [f"{materials[a].displayName}<->{materials[b].displayName}" for a, b in sorted(missing_pairs)]
        raise ValueError("missing material interface registration for " + ", ".join(names))
    unused_interfaces = np.flatnonzero(interface_face_counts == 0)
    if unused_interfaces.size:
        names = [str(int(identifier)) for identifier in unused_interfaces]
        raise ValueError("material interface registrations select no adjacent faces: " + ", ".join(names))

    return ResolvedProblem(
        mesh=mesh,
        materials=tuple(materials),
        cellMaterialId=cellMaterialId,
        boundaries=tuple(boundaries),
        faceBoundaryId=faceBoundaryId,
        interfaces=tuple(interfaces),
        faceInterfaceId=faceInterfaceId,
        initialExcitationFraction=_initial_state_array(mesh, initial_state),
    )
