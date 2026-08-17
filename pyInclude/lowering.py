# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lower the public physical graph to the current single-material backend."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hase_units import units
from .geometry import VolumeTopology
from .geometry.core import GainMedium as _BackendGainMedium
from .laser import CrossSectionData
from .physical import Domain, GainMedium, SURFACE, VOLUME


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
    occupied = {}
    for component in components:
        for topology, mask in component.domain._shards:
            previous = occupied.setdefault(id(topology), np.zeros_like(mask, dtype=bool))
            if np.any(previous & mask):
                raise ValueError("OpticalComponent volume domains must not overlap")
            previous |= mask
    material = gainMedium.components[0].material
    if any(component.material is not material for component in gainMedium.components[1:]):
        raise NotImplementedError(
            "the current backend requires all gain components to reference the same Material object"
        )
    material.validate()
    if not material.isActive:
        raise ValueError("GainMedium components require an active Material")
    if (
        material.bulkAttenuation is not None
        and float(material.bulkAttenuation.toValue(units.cm**-1)) != 0.0
    ):
        raise NotImplementedError(
            "the current backend does not support bulkAttenuation in active gain cells"
        )
    passive = tuple(component for component in components if component not in gainMedium.components)
    for component in passive:
        component.material.validate()
        if component.material.isActive:
            raise ValueError("OpticalComponents outside GainMedium must use passive Materials")
    return material, components, passive


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
                "the current backend supports Tet4 VolumeTopology domains only"
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
        metadata={"source": "OpticalComponent domain lowering"},
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


def _crossSections(material):
    table = material.crossSections
    if table is None:
        raise ValueError("active gain Material requires crossSections")
    return CrossSectionData(
        wavelengthsAbsorption=table.wavelengths.toValue(units.nm),
        crossSectionAbsorption=table.absorption.toValue(units.cm**2),
        wavelengthsEmission=table.wavelengths.toValue(units.nm),
        crossSectionEmission=table.emission.toValue(units.cm**2),
        resolution=int(table.wavelengths.size),
    )


def lowerSurfaceDomain(backendGainMedium, domain):
    """Assign a stable backend surface tag for one typed surface Domain."""
    selected = domain if isinstance(domain, Domain) else Domain(domain)
    if selected.entityKind != SURFACE or selected.isEmpty:
        raise ValueError("pump and optics selections require a non-empty surface Domain")
    cache = backendGainMedium.__dict__.setdefault("_surfaceDomainCache", {})
    key = tuple(
        (id(topology), tuple(mask.shape), np.packbits(mask.reshape(-1)).tobytes())
        for topology, mask in selected._shards
    )
    if key in cache:
        return cache[key]
    identifier = int(backendGainMedium.__dict__.setdefault("_nextSurfaceDomain", 1))
    backendGainMedium._nextSurfaceDomain = identifier + 1
    cell_maps = backendGainMedium.__dict__["_domainCellMaps"]
    for topology, faces in selected._shards:
        entry = cell_maps.get(id(topology))
        if entry is None or entry[0] is not topology:
            raise ValueError("surface Domain is outside the executable GainMedium")
        rows, local_faces = np.nonzero(faces)
        mapped = entry[1][rows]
        if np.any(mapped < 0):
            raise ValueError("surface Domain is outside the executable GainMedium")
        backendGainMedium.topology.faceBoundaries[mapped, local_faces] = identifier
    cache[key] = identifier
    return identifier


def lowerGainMedium(gainMedium, initialExcitation=0.0, *, opticalComponents=None):
    """Lower one active medium within its complete optical-component assembly.

    Public domains may use any compatible mesh representation. The current
    executable backend accepts Tet4 :class:`VolumeTopology` bindings only and
    rejects other discretizations here rather than in the physical graph.
    """
    material, components, passive = _validateComponents(gainMedium, opticalComponents)
    assembly_domain = Domain(component.domain for component in components)
    combined, cell_maps = _combineTopology(assembly_domain)
    excitation = _initialExcitation(
        initialExcitation,
        gainMedium.domain,
        cell_maps,
        combined.numberOfCells,
    )
    if material.fluorescenceLifetime is None:
        raise ValueError("active gain Material requires fluorescenceLifetime")
    backend = _BackendGainMedium(combined).withPhysicalProperties(
        betaVolume=excitation,
        nTot=float(material.activeIonDensity.toValue(units.cm**-3)),
        crystalTFluo=float(material.fluorescenceLifetime.toValue(units.s)),
    )
    backend._domainCellMaps = cell_maps
    backend._nextSurfaceDomain = int(np.max(combined.faceBoundaries, initial=0)) + 1

    if passive:
        attenuations = {
            0.0
            if component.material.bulkAttenuation is None
            else float(component.material.bulkAttenuation.toValue(units.cm**-1))
            for component in passive
        }
        if len(attenuations) != 1:
            raise NotImplementedError(
                "the current backend supports one passive bulkAttenuation value"
            )
        cell_types = np.zeros(combined.numberOfCells, dtype=np.uint32)
        for component in passive:
            cell_types[_indices(component.domain, cell_maps)] = np.uint32(1)
        backend.withPhysicalProperties(
            claddingCellTypes=cell_types,
            claddingNumber=1,
            claddingAbsorption=attenuations.pop(),
        )

    optics = {}
    for component in components:
        for selected, surfaceOptics in component.surfaceOptics:
            identifier = lowerSurfaceDomain(backend, selected)
            optics[identifier] = surfaceOptics
    if optics:
        backend.with_surface_optics(optics)
    return backend, _crossSections(material)


__all__ = ["lowerGainMedium", "lowerSurfaceDomain"]
