# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the Python-side state view used by callbacks and diagnostics.

Compiled execution does not consume this projection. The C++
``simulation preparation`` is the sole executable lowering of the transported
physical graph.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hase_units import units
from .geometry import VolumeTopology
from .geometry.core import GainMedium as _ProjectedGainMedium
from .physical import Domain, GainMedium, SURFACE, VOLUME, validateComponentOverlap


def _validateComponents(gainMedium, opticalComponents=None):
    if not isinstance(gainMedium, GainMedium) or not gainMedium.components:
        raise ValueError("Simulation requires a non-empty GainMedium")
    components = tuple(
        gainMedium.components if opticalComponents is None else opticalComponents
    )
    if not components:
        raise ValueError("Simulation requires at least one OpticalComponent")
    if any(component not in components for component in gainMedium.components):
        raise ValueError("every GainMedium component must belong to the optical assembly")
    validateComponentOverlap(components)
    for component in gainMedium.components:
        component.material.validate()
        if not component.material.isActive:
            raise ValueError("GainMedium components require active Materials")
        if float(component.material.activeIonDensity.toValue(units.cm**-3)) <= 0.0:
            raise ValueError("GainMedium components require positive activeIonDensity")
        if component.material.fluorescenceLifetime is None:
            raise ValueError("GainMedium components require fluorescenceLifetime")
        if component.material.crossSections is None:
            raise ValueError("GainMedium components require crossSections")
    passive = tuple(component for component in components if component not in gainMedium.components)
    for component in passive:
        component.material.validate()
        if component.material.isActive:
            raise ValueError("OpticalComponents outside GainMedium must use passive Materials")
    return components, passive


def _combineTopology(domain):
    points = []
    cells = []
    cell_domains = []
    face_boundaries = []
    cell_maps = {}
    global_cell = 0
    global_point = 0
    for topology, selected in domain._shards:
        if not isinstance(topology, VolumeTopology):
            raise NotImplementedError(
                "the callback state projection supports Tet4 VolumeTopology domains only"
            )
        selected_indices = np.flatnonzero(selected)
        source_cells = np.asarray(topology.cellPointIndices)[selected_indices]
        used_points = (
            np.arange(topology.numberOfPoints, dtype=np.int64)
            if np.all(selected)
            else np.unique(source_cells)
        )
        point_map = np.full(topology.numberOfPoints, -1, dtype=np.int64)
        point_map[used_points] = np.arange(used_points.size, dtype=np.int64) + global_point
        points.append(np.asarray(topology.points)[used_points])
        cells.append(point_map[source_cells])
        cell_domains.append(np.asarray(topology.cellDomains)[selected_indices])
        face_boundaries.append(np.asarray(topology.faceBoundaries)[selected_indices])
        mapping = np.full(topology.numberOfCells, -1, dtype=np.int64)
        mapping[selected_indices] = np.arange(selected_indices.size, dtype=np.int64) + global_cell
        cell_maps[id(topology)] = (topology, mapping)
        global_cell += selected_indices.size
        global_point += used_points.size
    if not cells:
        raise ValueError("optical assembly domain must not be empty")
    combined = VolumeTopology.fromTetrahedra(
        np.concatenate(points),
        np.concatenate(cells),
        cellDomains=np.concatenate(cell_domains),
        faceBoundaries=np.concatenate(face_boundaries),
        metadata={"source": "OpticalComponent frontend state projection"},
    )
    return combined, cell_maps


def _indices(domain, cellMaps):
    if not isinstance(domain, Domain) or domain.entityKind != VOLUME:
        raise TypeError("cell assignments require volume Domains")
    result = []
    for topology, mask in domain._shards:
        entry = cellMaps.get(id(topology))
        if entry is None or entry[0] is not topology:
            if np.any(mask):
                raise ValueError("Domain contains cells outside the optical assembly")
            continue
        mapped = entry[1][mask]
        if np.any(mapped < 0):
            raise ValueError("Domain contains cells outside the optical assembly")
        result.extend(mapped.tolist())
    return np.asarray(result, dtype=np.int64)


def _initialExcitation(value, domain, cellMaps, numberOfCells):
    active_indices = _indices(domain, cellMaps)
    active = np.zeros(numberOfCells, dtype=bool)
    active[active_indices] = True
    result = np.zeros(numberOfCells, dtype=np.float64)
    if isinstance(value, Mapping):
        covered = np.zeros(numberOfCells, dtype=bool)
        for selected, assigned in value.items():
            indices = _indices(selected, cellMaps)
            if np.any(~active[indices]):
                raise ValueError("initial excitation Domains must stay inside the GainMedium")
            if np.any(covered[indices]):
                raise ValueError("initial excitation Domains overlap")
            values = np.asarray(assigned, dtype=np.float64)
            if values.ndim == 0:
                values = np.full(indices.size, float(values), dtype=np.float64)
            else:
                values = values.reshape(-1)
                if values.size != indices.size:
                    raise ValueError("initial excitation array must match its Domain size")
            result[indices] = values
            covered[indices] = True
        if not np.all(covered[active_indices]):
            raise ValueError("initial excitation Domains must cover the GainMedium exactly once")
    else:
        result[active_indices] = float(value)
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("initial excitation must be finite and within [0, 1]")
    return result


def projectSurfaceDomain(projectedState, domain):
    """Assign a stable diagnostic surface tag in the frontend state view."""
    selected = domain if isinstance(domain, Domain) else Domain(domain)
    if selected.entityKind != SURFACE or selected.isEmpty:
        raise ValueError("pump and optics selections require a non-empty surface Domain")
    cache = projectedState.__dict__.setdefault("_surfaceDomainCache", {})
    key = tuple(
        (id(topology), tuple(mask.shape), np.packbits(mask.reshape(-1)).tobytes())
        for topology, mask in selected._shards
    )
    if key in cache:
        return cache[key]
    identifier = int(projectedState.__dict__.setdefault("_nextSurfaceDomain", 1))
    projectedState._nextSurfaceDomain = identifier + 1
    cell_maps = projectedState.__dict__["_domainCellMaps"]
    for topology, faces in selected._shards:
        entry = cell_maps.get(id(topology))
        if entry is None or entry[0] is not topology:
            raise ValueError("surface Domain is outside the frontend state view")
        rows, local_faces = np.nonzero(faces)
        mapped = entry[1][rows]
        if np.any(mapped < 0):
            raise ValueError("surface Domain is outside the frontend state view")
        projectedState.topology.faceBoundaries[mapped, local_faces] = identifier
    cache[key] = identifier
    return identifier


def projectCellMask(projectedState, domain):
    """Return a callback-state mask for one physical volume ``Domain``.

    The mapping follows the prepared callback cell order and does not copy or
    lower any material properties.
    """
    result = np.zeros(projectedState.topology.numberOfCells, dtype=bool)
    result[_indices(domain, projectedState.__dict__["_domainCellMaps"])] = True
    return result


def projectFrontendState(gainMedium, initialExcitation=0.0, *, opticalComponents=None):
    """Project one assembly for Python callbacks and diagnostics.

    Public domains may use any compatible mesh representation. The current
    frontend state containers represent Tet4 :class:`VolumeTopology` bindings
    only. Runtime validation and physical lowering remain owned by C++.
    """
    components, _passive = _validateComponents(gainMedium, opticalComponents)
    assembly_domain = Domain(component.domain for component in components)
    combined, cell_maps = _combineTopology(assembly_domain)
    excitation = _initialExcitation(
        initialExcitation,
        gainMedium.domain,
        cell_maps,
        combined.numberOfCells,
    )
    state = _ProjectedGainMedium(combined).withPhysicalProperties(betaVolume=excitation)
    state._domainCellMaps = cell_maps
    state._nextSurfaceDomain = int(np.max(combined.faceBoundaries, initial=0)) + 1

    optics = {}
    for component in components:
        for selected, surfaceOptics in component.surfaceOptics:
            identifier = projectSurfaceDomain(state, selected)
            optics[identifier] = surfaceOptics
    if optics:
        state.with_surface_optics(optics)
    return state


__all__ = ["projectCellMask", "projectFrontendState", "projectSurfaceDomain"]
