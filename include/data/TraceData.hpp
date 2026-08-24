/**
 * Copyright 2013 Erik Zenker, Carlchristian Eckert, Marius Melzer
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @author Erik Zenker
 * @author Carlchristian Eckert
 * @author Marius Melzer
 * @licence GPLv3
 *
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/core/common.hpp>

#include <alpakaUtils/HybridBuffer.hpp>
#include <alpakaUtils/utils.hpp>
#include <core/geometry.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>
#define REFLECTION_SMALL 1E-3
#define SMALL 1E-5
#define VERY_SMALL 0.0

namespace hase::data
{
    using core::cross;
    using core::dot;
    using core::Point;
    using core::TwoDimPoint;
    constexpr unsigned tet4VertexCount = 4u;
    constexpr unsigned tet4FaceCount = 4u;
    constexpr unsigned tet4FaceWidth = 3u;
    constexpr unsigned tet4BarycentricPlaneWidth = 4u;
    constexpr unsigned vtkTetraCellType = 10u;

    template<class T, class B, class E>
    inline void assertRange(
        [[maybe_unused]] std::vector<T> const& v,
        [[maybe_unused]] B const minElement,
        [[maybe_unused]] E const maxElement,
        [[maybe_unused]] bool const equals)
    {
        if(equals)
        {
            assert(*std::min_element(v.begin(), v.end()) == minElement);
            assert(*std::max_element(v.begin(), v.end()) == maxElement);
        }
        else
        {
            assert(*std::min_element(v.begin(), v.end()) >= minElement);
            assert(*std::max_element(v.begin(), v.end()) <= maxElement);
        }
    }

    template<class T, class B>
    inline void assertMin(
        [[maybe_unused]] std::vector<T> const& v,
        [[maybe_unused]] B const minElement,
        [[maybe_unused]] bool const equals)
    {
        if(equals)
        {
            assert(*std::min_element(v.begin(), v.end()) == minElement);
        }
        else
        {
            assert(*std::min_element(v.begin(), v.end()) >= minElement);
        }
    }

    inline double distance2D(TwoDimPoint const p1, TwoDimPoint const p2)
    {
        return std::abs(std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)));
    }

    inline double getMaxDistance(std::vector<TwoDimPoint> const& points)
    {
        double maxDistance = -1.0;

        for(unsigned p1 = 0; p1 < points.size(); ++p1)
        {
            for(unsigned p2 = p1; p2 < points.size(); ++p2)
            {
                maxDistance = std::max(maxDistance, distance2D(points[p1], points[p2]));
            }
        }

        return maxDistance;
    }

    inline double calculateMaxDiameter(double const* points, unsigned const offset)
    {
        TwoDimPoint minX = {std::numeric_limits<double>::max(), 0};
        TwoDimPoint minY = {0, std::numeric_limits<double>::max()};
        TwoDimPoint maxX = {std::numeric_limits<double>::lowest(), 0};
        TwoDimPoint maxY = {0, std::numeric_limits<double>::lowest()};

        for(unsigned p = 0; p < offset; ++p)
        {
            TwoDimPoint np = {points[p], points[p + offset]};
            minX = (points[p] < minX.x) ? np : minX;
            maxX = (points[p] > maxX.x) ? np : maxX;
        }

        for(unsigned p = offset; p < 2 * offset; ++p)
        {
            TwoDimPoint np = {points[p - offset], points[p]};
            minY = points[p] < minY.y ? np : minY;
            maxY = points[p] > maxY.y ? np : maxY;
        }

        std::vector<TwoDimPoint> extrema;
        extrema.push_back(minX);
        extrema.push_back(minY);
        extrema.push_back(maxX);
        extrema.push_back(maxY);

        return getMaxDistance(extrema);
    }

    /**
     * @brief Trivially-copyable, non-owning view consumed by ray kernels.
     *
     * Material properties are indexed through cellMaterialIds. Rays carry a
     * wavelength and resolve cross sections for every receiving cell, so a
     * trace has no process-global active material or spectral array.
     */
    struct TraceView
    {
        struct CrossSections
        {
            double absorption = 0.0;
            double emission = 0.0;
        };

        std::span<double const> points;
        std::span<double const> betaVolume;
        std::span<unsigned const> cellMaterialIds;
        std::span<std::uint8_t const> materialActive;
        std::span<double const> materialRefractiveIndices;
        std::span<double const> materialActiveIonDensities;
        std::span<double const> materialFluorescenceLifetimes;
        std::span<double const> materialBulkAttenuations;
        std::span<double const> materialPeakAbsorption;
        std::span<double const> materialPeakEmission;
        std::span<unsigned const> materialCrossSectionOffsets;
        std::span<double const> crossSectionWavelengths;
        std::span<double const> crossSectionAbsorption;
        std::span<double const> crossSectionEmission;
        std::span<float const> surfaceReflectivities;
        std::span<float const> surfaceRefractiveIndexInside;
        std::span<float const> surfaceRefractiveIndexOutside;
        std::span<unsigned const> cellPointIndices;
        std::span<unsigned const> cellTypes;
        std::span<int const> cellFaces;
        // Affine barycentric coordinates of the vertex opposite each local face.
        std::span<double const> barycentricFacePlanes;
        std::span<int const> cellNeighborCells;
        std::span<int const> cellNeighborLocalFaces;
        std::span<int const> cellFaceBoundaries;
        std::span<double const> cellVolumes;
        std::span<double const> lumpedMaterialVertexVolumes;
        std::span<double const> cellVolumePrefix;
        std::span<double const> sourceStrengthPrefix;
        std::span<double const> cellCenters;
        std::span<double const> samplePoints;

        unsigned numberOfMaterials;
        unsigned numberOfCells;
        unsigned numberOfPoints;
        unsigned numberOfSamples;
        unsigned numberOfFacesPerCell;
        unsigned numberOfCellVertices;
        unsigned numberOfMeshPoints;
        unsigned numberOfLevels;
        float thickness;
        bool samplePointsAreMeshPoints;

        /** @param pointIndex Global mesh-point index. @return Point coordinates in Cartesian layout. */
        [[nodiscard]] ALPAKA_FN_ACC Point getPoint(unsigned pointIndex) const
        {
            return Point{
                points[pointIndex],
                points[pointIndex + numberOfMeshPoints],
                points[pointIndex + 2u * numberOfMeshPoints]};
        }

        /** @param cell Cell index. @param localVertex Cell-local vertex index. @return Referenced mesh point. */
        [[nodiscard]] ALPAKA_FN_ACC Point getCellPoint(unsigned cell, unsigned localVertex) const
        {
            return getPoint(cellPointIndices[cell * numberOfCellVertices + localVertex]);
        }

        /**
         * @param cell Cell index.
         * @param localFace Cell-local face index.
         * @param localVertex Face-local vertex index.
         * @return Global mesh-point index, or the stored boundary sentinel.
         */
        [[nodiscard]] ALPAKA_FN_ACC int getCellFacePoint(unsigned cell, unsigned localFace, unsigned localVertex) const
        {
            return cellFaces[(cell * numberOfFacesPerCell + localFace) * tet4FaceWidth + localVertex];
        }

        /**
         * @param cell Cell index.
         * @param localFace Cell-local face index.
         * @param point Point at which to evaluate the opposite-vertex coordinate.
         * @return Affine barycentric face coordinate; zero lies on the face.
         */
        [[nodiscard]] ALPAKA_FN_ACC double getFaceBarycentricCoordinate(
            unsigned const cell,
            unsigned const localFace,
            Point const point) const
        {
            unsigned const offset = (cell * numberOfFacesPerCell + localFace) * tet4BarycentricPlaneWidth;
            return barycentricFacePlanes[offset] * point.x + barycentricFacePlanes[offset + 1u] * point.y
                   + barycentricFacePlanes[offset + 2u] * point.z + barycentricFacePlanes[offset + 3u];
        }

        /**
         * @param cell Cell index.
         * @param localFace Cell-local face index.
         * @param direction Direction to project onto the face-coordinate gradient.
         * @return Directional derivative of the barycentric face coordinate.
         */
        [[nodiscard]] ALPAKA_FN_ACC double getFaceBarycentricDirection(
            unsigned const cell,
            unsigned const localFace,
            Point const direction) const
        {
            unsigned const offset = (cell * numberOfFacesPerCell + localFace) * tet4BarycentricPlaneWidth;
            return barycentricFacePlanes[offset] * direction.x + barycentricFacePlanes[offset + 1u] * direction.y
                   + barycentricFacePlanes[offset + 2u] * direction.z;
        }

        /** @param cell Cell index. @param localFace Cell-local face index. @return Neighbor cell or a boundary
         * sentinel. */
        [[nodiscard]] ALPAKA_FN_ACC int getCellNeighbor(unsigned cell, unsigned localFace) const
        {
            return cellNeighborCells[cell * numberOfFacesPerCell + localFace];
        }

        /** @param cell Cell index. @param localFace Cell-local face index. @return Matching local face in the
         * neighbor. */
        [[nodiscard]] ALPAKA_FN_ACC int getCellNeighborLocalFace(unsigned cell, unsigned localFace) const
        {
            return cellNeighborLocalFaces[cell * numberOfFacesPerCell + localFace];
        }

        /** @param cell Cell index. @return Excited-state fraction stored for the cell. */
        [[nodiscard]] ALPAKA_FN_ACC double getBetaVolume(unsigned cell) const
        {
            return betaVolume[cell];
        }

        /** @param cell Cell index. @return Index of the cell's material. */
        [[nodiscard]] ALPAKA_FN_ACC unsigned getMaterialId(unsigned cell) const
        {
            return cellMaterialIds[cell];
        }

        /** @param cell Cell index. @return Whether the assigned material contributes gain. */
        [[nodiscard]] ALPAKA_FN_ACC bool isActive(unsigned cell) const
        {
            return materialActive[getMaterialId(cell)] != 0u;
        }

        /** @param cell Cell index. @return Active-ion number density of the assigned material. */
        [[nodiscard]] ALPAKA_FN_ACC double activeIonDensity(unsigned cell) const
        {
            return materialActiveIonDensities[getMaterialId(cell)];
        }

        /** @param cell Cell index. @return Fluorescence lifetime of the assigned material. */
        [[nodiscard]] ALPAKA_FN_ACC double fluorescenceLifetime(unsigned cell) const
        {
            return materialFluorescenceLifetimes[getMaterialId(cell)];
        }

        /** @param cell Cell index. @return Bulk attenuation coefficient of the assigned material. */
        [[nodiscard]] ALPAKA_FN_ACC double bulkAttenuation(unsigned cell) const
        {
            return materialBulkAttenuations[getMaterialId(cell)];
        }

        /** @param material Material index. @return Number of spectral samples owned by the material. */
        [[nodiscard]] ALPAKA_FN_ACC unsigned crossSectionCount(unsigned material) const
        {
            return materialCrossSectionOffsets[material + 1u] - materialCrossSectionOffsets[material];
        }

        /** @param material Material index. @param sample Material-local spectral index. @return Sample wavelength. */
        [[nodiscard]] ALPAKA_FN_ACC double emissionWavelength(unsigned material, unsigned sample) const
        {
            return crossSectionWavelengths[materialCrossSectionOffsets[material] + sample];
        }

        /**
         * @param material Material index.
         * @param wavelength Wavelength at which to interpolate, clamped to the table endpoints.
         * @return Absorption and emission cross sections for the material.
         */
        [[nodiscard]] ALPAKA_FN_ACC CrossSections crossSections(unsigned material, double wavelength) const
        {
            unsigned const begin = materialCrossSectionOffsets[material];
            unsigned const end = materialCrossSectionOffsets[material + 1u];
            if(begin == end)
                return {};
            if(end - begin == 1u || wavelength <= crossSectionWavelengths[begin])
                return {crossSectionAbsorption[begin], crossSectionEmission[begin]};
            if(wavelength >= crossSectionWavelengths[end - 1u])
                return {crossSectionAbsorption[end - 1u], crossSectionEmission[end - 1u]};

            unsigned lower = begin;
            unsigned upper = end;
            while(lower < upper)
            {
                unsigned const middle = lower + (upper - lower) / 2u;
                if(crossSectionWavelengths[middle] <= wavelength)
                    lower = middle + 1u;
                else
                    upper = middle;
            }
            unsigned const right = lower;
            unsigned const left = right - 1u;
            double const fraction = (wavelength - crossSectionWavelengths[left])
                                    / (crossSectionWavelengths[right] - crossSectionWavelengths[left]);
            return {
                crossSectionAbsorption[left]
                    + fraction * (crossSectionAbsorption[right] - crossSectionAbsorption[left]),
                crossSectionEmission[left] + fraction * (crossSectionEmission[right] - crossSectionEmission[left])};
        }

        /** @param cell Cell index. @param wavelength Query wavelength. @return Cross sections of the cell's material.
         */
        [[nodiscard]] ALPAKA_FN_ACC CrossSections crossSectionsForCell(unsigned cell, double wavelength) const
        {
            return crossSections(getMaterialId(cell), wavelength);
        }

        /** @param a First vertex. @param b Second vertex. @param c Third vertex. @param d Fourth vertex. @return
         * Tetrahedron volume. */
        [[nodiscard]] ALPAKA_FN_ACC double tetraVolume(Point const a, Point const b, Point const c, Point const d)
            const
        {
            return alpaka::math::abs(dot(cross(b - a, c - a), d - a)) / 6.0;
        }

        /**
         * @param a First vertex. @param b Second vertex. @param c Third vertex. @param d Fourth vertex.
         * @param rndEngine Random engine advanced by the sampling operation.
         * @return A uniformly distributed point inside the tetrahedron.
         */
        ALPAKA_FN_ACC Point genRndPointInTetra(
            Point const a,
            Point const b,
            Point const c,
            Point const d,
            alpaka::rand::concepts::UniformRandomEngine auto& rndEngine) const
        {
            double r0 = alpaka::rand::distribution::UniformReal<double>{}(rndEngine);
            double r1 = alpaka::rand::distribution::UniformReal<double>{}(rndEngine);
            double r2 = alpaka::rand::distribution::UniformReal<double>{}(rndEngine);
            double r3 = alpaka::rand::distribution::UniformReal<double>{}(rndEngine);
            r0 = -alpaka::math::log(alpaka::math::max(r0, std::numeric_limits<double>::min()));
            r1 = -alpaka::math::log(alpaka::math::max(r1, std::numeric_limits<double>::min()));
            r2 = -alpaka::math::log(alpaka::math::max(r2, std::numeric_limits<double>::min()));
            r3 = -alpaka::math::log(alpaka::math::max(r3, std::numeric_limits<double>::min()));
            double const invSum = 1.0 / (r0 + r1 + r2 + r3);
            return a * (r0 * invSum) + b * (r1 * invSum) + c * (r2 * invSum) + d * (r3 * invSum);
        }

        /**
         * @param origin Point that the sample must not coincide with.
         * @param cell Cell whose tetrahedron is sampled.
         * @param rndEngine Random engine advanced by the sampling operation.
         * @return A uniformly distributed point inside the cell distinct from `origin`.
         */
        ALPAKA_FN_ACC Point genRndPointInCell(
            Point& origin,
            unsigned cell,
            alpaka::rand::concepts::UniformRandomEngine auto& rndEngine) const
        {
            Point const p0 = getCellPoint(cell, 0u);
            Point const p1 = getCellPoint(cell, 1u);
            Point const p2 = getCellPoint(cell, 2u);
            Point const p3 = getCellPoint(cell, 3u);

            Point startPoint = genRndPointInTetra(p0, p1, p2, p3, rndEngine);
            if((origin - startPoint).euclidLength() < SMALL)
            {
                return genRndPointInCell(origin, cell, rndEngine);
            }
            return startPoint;
        }

        /** @param sampleIndex Sample index. @return Sample coordinates from the structure-of-arrays layout. */
        [[nodiscard]] ALPAKA_FN_ACC Point getSamplePoint(unsigned sampleIndex) const
        {
            return Point{
                samplePoints[sampleIndex],
                samplePoints[sampleIndex + numberOfSamples],
                samplePoints[sampleIndex + 2u * numberOfSamples]};
        }

        /** @param cell Cell index. @return Precomputed cell center. */
        [[nodiscard]] ALPAKA_FN_ACC Point getCellCenterPoint(unsigned cell) const
        {
            return Point{cellCenters[cell], cellCenters[cell + numberOfCells], cellCenters[cell + 2u * numberOfCells]};
        }

        /** @param cell Cell index. @return Physical cell volume. */
        [[nodiscard]] ALPAKA_FN_ACC double getCellVolume(unsigned cell) const
        {
            return cellVolumes[cell];
        }

        /** @param cell Cell index. @param localFace Cell-local face index. @return Assigned constant reflectivity, or
         * zero. */
        [[nodiscard]] ALPAKA_FN_ACC float getSurfaceReflectivity(unsigned cell, unsigned localFace) const
        {
            int const surfaceId = cellFaceBoundaries[cell * numberOfFacesPerCell + localFace];
            if(surfaceId > 0 && static_cast<unsigned>(surfaceId) < surfaceReflectivities.size())
            {
                return surfaceReflectivities[static_cast<unsigned>(surfaceId)];
            }
            return 0.0f;
        }

        /** @param cell Cell index. @param localFace Cell-local face index. @return Refractive index on the incident
         * side. */
        [[nodiscard]] ALPAKA_FN_ACC float getSurfaceRefractiveIndexInside(unsigned cell, unsigned localFace) const
        {
            int const surfaceId = cellFaceBoundaries[cell * numberOfFacesPerCell + localFace];
            if(surfaceId > 0 && static_cast<unsigned>(surfaceId) < surfaceRefractiveIndexInside.size())
            {
                return surfaceRefractiveIndexInside[static_cast<unsigned>(surfaceId)];
            }
            unsigned const material = getMaterialId(cell);
            return material < materialRefractiveIndices.size()
                       ? static_cast<float>(materialRefractiveIndices[material])
                       : 1.0f;
        }

        /** @param cell Cell index. @param localFace Cell-local face index. @return Refractive index on the exterior
         * side. */
        [[nodiscard]] ALPAKA_FN_ACC float getSurfaceRefractiveIndexOutside(unsigned cell, unsigned localFace) const
        {
            int const surfaceId = cellFaceBoundaries[cell * numberOfFacesPerCell + localFace];
            if(surfaceId > 0 && static_cast<unsigned>(surfaceId) < surfaceRefractiveIndexOutside.size())
            {
                return surfaceRefractiveIndexOutside[static_cast<unsigned>(surfaceId)];
            }
            return 1.0f;
        }
    };

    class TraceData;

    /**
     * @brief HybridBuffer ownership for one device-local tracing domain.
     *
     * Geometry, materials, spectra, source prefixes, and excitation stay
     * resident across launches. A future multi-domain scheduler can own one
     * instance per OpticalComponent and exchange only boundary-ray records
     * between iterations; kernels remain local to one TraceView.
     */
    template<alpaka::onHost::concepts::Device T_Device>
    class ResidentTrace
    {
    public:
        ResidentTrace(
            T_Device device,
            unsigned numberOfMaterials,
            unsigned numberOfCells,
            unsigned numberOfPoints,
            unsigned numberOfSamples,
            unsigned numberOfFacesPerCell,
            unsigned numberOfCellVertices,
            unsigned numberOfMeshPoints,
            unsigned numberOfLevels,
            float thickness,
            bool samplePointsAreMeshPoints,
            std::vector<double>& points,
            std::vector<double>& betaVolume,
            std::vector<unsigned>& cellMaterialIds,
            std::vector<std::uint8_t>& materialActive,
            std::vector<double>& materialRefractiveIndices,
            std::vector<double>& materialActiveIonDensities,
            std::vector<double>& materialFluorescenceLifetimes,
            std::vector<double>& materialBulkAttenuations,
            std::vector<double>& materialPeakAbsorption,
            std::vector<double>& materialPeakEmission,
            std::vector<unsigned>& materialCrossSectionOffsets,
            std::vector<double>& crossSectionWavelengths,
            std::vector<double>& crossSectionAbsorption,
            std::vector<double>& crossSectionEmission,
            std::vector<float>& surfaceReflectivities,
            std::vector<float>& surfaceRefractiveIndexInside,
            std::vector<float>& surfaceRefractiveIndexOutside,
            std::vector<unsigned>& cellPointIndices,
            std::vector<unsigned>& cellTypes,
            std::vector<int>& cellFaces,
            std::vector<double>& barycentricFacePlanes,
            std::vector<int>& cellNeighborCells,
            std::vector<int>& cellNeighborLocalFaces,
            std::vector<int>& cellFaceBoundaries,
            std::vector<double>& cellVolumes,
            std::vector<double>& lumpedMaterialVertexVolumes,
            std::vector<double>& cellVolumePrefix,
            std::vector<double>& sourceStrengthPrefix,
            std::vector<double>& cellCenters,
            std::vector<double>& samplePoints)
            : m_device(device)
            , points(hase::alpakaUtils::getHybridBuffer(m_device, points))
            , betaVolume(hase::alpakaUtils::getHybridBuffer(m_device, betaVolume))
            , cellMaterialIds(hase::alpakaUtils::getHybridBuffer(m_device, cellMaterialIds))
            , materialActive(hase::alpakaUtils::getHybridBuffer(m_device, materialActive))
            , materialRefractiveIndices(hase::alpakaUtils::getHybridBuffer(m_device, materialRefractiveIndices))
            , materialActiveIonDensities(hase::alpakaUtils::getHybridBuffer(m_device, materialActiveIonDensities))
            , materialFluorescenceLifetimes(
                  hase::alpakaUtils::getHybridBuffer(m_device, materialFluorescenceLifetimes))
            , materialBulkAttenuations(hase::alpakaUtils::getHybridBuffer(m_device, materialBulkAttenuations))
            , materialPeakAbsorption(hase::alpakaUtils::getHybridBuffer(m_device, materialPeakAbsorption))
            , materialPeakEmission(hase::alpakaUtils::getHybridBuffer(m_device, materialPeakEmission))
            , materialCrossSectionOffsets(hase::alpakaUtils::getHybridBuffer(m_device, materialCrossSectionOffsets))
            , crossSectionWavelengths(hase::alpakaUtils::getHybridBuffer(m_device, crossSectionWavelengths))
            , crossSectionAbsorption(hase::alpakaUtils::getHybridBuffer(m_device, crossSectionAbsorption))
            , crossSectionEmission(hase::alpakaUtils::getHybridBuffer(m_device, crossSectionEmission))
            , surfaceReflectivities(hase::alpakaUtils::getHybridBuffer(m_device, surfaceReflectivities))
            , surfaceRefractiveIndexInside(hase::alpakaUtils::getHybridBuffer(m_device, surfaceRefractiveIndexInside))
            , surfaceRefractiveIndexOutside(
                  hase::alpakaUtils::getHybridBuffer(m_device, surfaceRefractiveIndexOutside))
            , cellPointIndices(hase::alpakaUtils::getHybridBuffer(m_device, cellPointIndices))
            , cellTypes(hase::alpakaUtils::getHybridBuffer(m_device, cellTypes))
            , cellFaces(hase::alpakaUtils::getHybridBuffer(m_device, cellFaces))
            , barycentricFacePlanes(hase::alpakaUtils::getHybridBuffer(m_device, barycentricFacePlanes))
            , cellNeighborCells(hase::alpakaUtils::getHybridBuffer(m_device, cellNeighborCells))
            , cellNeighborLocalFaces(hase::alpakaUtils::getHybridBuffer(m_device, cellNeighborLocalFaces))
            , cellFaceBoundaries(hase::alpakaUtils::getHybridBuffer(m_device, cellFaceBoundaries))
            , cellVolumes(hase::alpakaUtils::getHybridBuffer(m_device, cellVolumes))
            , lumpedMaterialVertexVolumes(hase::alpakaUtils::getHybridBuffer(m_device, lumpedMaterialVertexVolumes))
            , cellVolumePrefix(hase::alpakaUtils::getHybridBuffer(m_device, cellVolumePrefix))
            , sourceStrengthPrefix(hase::alpakaUtils::getHybridBuffer(m_device, sourceStrengthPrefix))
            , cellCenters(hase::alpakaUtils::getHybridBuffer(m_device, cellCenters))
            , samplePoints(hase::alpakaUtils::getHybridBuffer(m_device, samplePoints))
            , numberOfMaterials(numberOfMaterials)
            , numberOfCells(numberOfCells)
            , numberOfPoints(numberOfPoints)
            , numberOfSamples(numberOfSamples)
            , numberOfFacesPerCell(numberOfFacesPerCell)
            , numberOfCellVertices(numberOfCellVertices)
            , numberOfMeshPoints(numberOfMeshPoints)
            , numberOfLevels(numberOfLevels)
            , thickness(thickness)
            , samplePointsAreMeshPoints(samplePointsAreMeshPoints)
        {
        }

        /**
         * @brief Enqueue host-to-device copies for every trace buffer.
         * @param queue Queue associated with `m_device` on which copies are enqueued.
         * @note The function does not wait; callers synchronize the queue before kernel use.
         */
        void toDevice(concepts::Queue auto const& queue)
        {
            points.toDevice(queue);
            betaVolume.toDevice(queue);
            cellMaterialIds.toDevice(queue);
            materialActive.toDevice(queue);
            materialRefractiveIndices.toDevice(queue);
            materialActiveIonDensities.toDevice(queue);
            materialFluorescenceLifetimes.toDevice(queue);
            materialBulkAttenuations.toDevice(queue);
            materialPeakAbsorption.toDevice(queue);
            materialPeakEmission.toDevice(queue);
            materialCrossSectionOffsets.toDevice(queue);
            crossSectionWavelengths.toDevice(queue);
            crossSectionAbsorption.toDevice(queue);
            crossSectionEmission.toDevice(queue);
            surfaceReflectivities.toDevice(queue);
            surfaceRefractiveIndexInside.toDevice(queue);
            surfaceRefractiveIndexOutside.toDevice(queue);
            cellPointIndices.toDevice(queue);
            cellTypes.toDevice(queue);
            cellFaces.toDevice(queue);
            barycentricFacePlanes.toDevice(queue);
            cellNeighborCells.toDevice(queue);
            cellNeighborLocalFaces.toDevice(queue);
            cellFaceBoundaries.toDevice(queue);
            cellVolumes.toDevice(queue);
            lumpedMaterialVertexVolumes.toDevice(queue);
            cellVolumePrefix.toDevice(queue);
            sourceStrengthPrefix.toDevice(queue);
            cellCenters.toDevice(queue);
            samplePoints.toDevice(queue);
        }

        /**
         * @brief Replace and upload only material-owned arrays.
         *
         * Topology buffers remain allocated and resident. This is the explicit
         * synchronization path for resized cross-section tables.
         * @param trace Host trace providing replacement material arrays.
         * @param queue Queue associated with `m_device` on which uploads are enqueued.
         * @note The function does not wait for the uploads to complete.
         */
        void refreshMaterials(TraceData& trace, concepts::Queue auto const& queue);

        /**
         * @brief Build the non-owning device view passed to tracing kernels.
         * @return Spans into the current device allocations plus their layout metadata.
         * @warning The view is invalidated when any referenced HybridBuffer is replaced.
         */
        [[nodiscard]] auto view() const -> TraceView
        {
            return {
                std::span<double const>(points.toDeviceView().data(), points.getExtents().x()),
                std::span<double const>(betaVolume.toDeviceView().data(), betaVolume.getExtents().x()),
                std::span<unsigned const>(cellMaterialIds.toDeviceView().data(), cellMaterialIds.getExtents().x()),
                std::span<std::uint8_t const>(materialActive.toDeviceView().data(), materialActive.getExtents().x()),
                std::span<double const>(
                    materialRefractiveIndices.toDeviceView().data(),
                    materialRefractiveIndices.getExtents().x()),
                std::span<double const>(
                    materialActiveIonDensities.toDeviceView().data(),
                    materialActiveIonDensities.getExtents().x()),
                std::span<double const>(
                    materialFluorescenceLifetimes.toDeviceView().data(),
                    materialFluorescenceLifetimes.getExtents().x()),
                std::span<double const>(
                    materialBulkAttenuations.toDeviceView().data(),
                    materialBulkAttenuations.getExtents().x()),
                std::span<double const>(
                    materialPeakAbsorption.toDeviceView().data(),
                    materialPeakAbsorption.getExtents().x()),
                std::span<double const>(
                    materialPeakEmission.toDeviceView().data(),
                    materialPeakEmission.getExtents().x()),
                std::span<unsigned const>(
                    materialCrossSectionOffsets.toDeviceView().data(),
                    materialCrossSectionOffsets.getExtents().x()),
                std::span<double const>(
                    crossSectionWavelengths.toDeviceView().data(),
                    crossSectionWavelengths.getExtents().x()),
                std::span<double const>(
                    crossSectionAbsorption.toDeviceView().data(),
                    crossSectionAbsorption.getExtents().x()),
                std::span<double const>(
                    crossSectionEmission.toDeviceView().data(),
                    crossSectionEmission.getExtents().x()),
                std::span<float const>(
                    surfaceReflectivities.toDeviceView().data(),
                    surfaceReflectivities.getExtents().x()),
                std::span<float const>(
                    surfaceRefractiveIndexInside.toDeviceView().data(),
                    surfaceRefractiveIndexInside.getExtents().x()),
                std::span<float const>(
                    surfaceRefractiveIndexOutside.toDeviceView().data(),
                    surfaceRefractiveIndexOutside.getExtents().x()),
                std::span<unsigned const>(cellPointIndices.toDeviceView().data(), cellPointIndices.getExtents().x()),
                std::span<unsigned const>(cellTypes.toDeviceView().data(), cellTypes.getExtents().x()),
                std::span<int const>(cellFaces.toDeviceView().data(), cellFaces.getExtents().x()),
                std::span<double const>(
                    barycentricFacePlanes.toDeviceView().data(),
                    barycentricFacePlanes.getExtents().x()),
                std::span<int const>(cellNeighborCells.toDeviceView().data(), cellNeighborCells.getExtents().x()),
                std::span<int const>(
                    cellNeighborLocalFaces.toDeviceView().data(),
                    cellNeighborLocalFaces.getExtents().x()),
                std::span<int const>(cellFaceBoundaries.toDeviceView().data(), cellFaceBoundaries.getExtents().x()),
                std::span<double const>(cellVolumes.toDeviceView().data(), cellVolumes.getExtents().x()),
                std::span<double const>(
                    lumpedMaterialVertexVolumes.toDeviceView().data(),
                    lumpedMaterialVertexVolumes.getExtents().x()),
                std::span<double const>(cellVolumePrefix.toDeviceView().data(), cellVolumePrefix.getExtents().x()),
                std::span<double const>(
                    sourceStrengthPrefix.toDeviceView().data(),
                    sourceStrengthPrefix.getExtents().x()),
                std::span<double const>(cellCenters.toDeviceView().data(), cellCenters.getExtents().x()),
                std::span<double const>(samplePoints.toDeviceView().data(), samplePoints.getExtents().x()),
                numberOfMaterials,
                numberOfCells,
                numberOfPoints,
                numberOfSamples,
                numberOfFacesPerCell,
                numberOfCellVertices,
                numberOfMeshPoints,
                numberOfLevels,
                thickness,
                samplePointsAreMeshPoints};
        }

        T_Device m_device;

    public:
        template<typename T_Data>
        using T_Buffer = hase::alpakaUtils::GetHybridBuffer_t<T_Device, std::vector<T_Data>>;

        T_Buffer<double> points;
        T_Buffer<double> betaVolume;
        T_Buffer<unsigned> cellMaterialIds;
        T_Buffer<std::uint8_t> materialActive;
        T_Buffer<double> materialRefractiveIndices;
        T_Buffer<double> materialActiveIonDensities;
        T_Buffer<double> materialFluorescenceLifetimes;
        T_Buffer<double> materialBulkAttenuations;
        T_Buffer<double> materialPeakAbsorption;
        T_Buffer<double> materialPeakEmission;
        T_Buffer<unsigned> materialCrossSectionOffsets;
        T_Buffer<double> crossSectionWavelengths;
        T_Buffer<double> crossSectionAbsorption;
        T_Buffer<double> crossSectionEmission;
        T_Buffer<float> surfaceReflectivities;
        T_Buffer<float> surfaceRefractiveIndexInside;
        T_Buffer<float> surfaceRefractiveIndexOutside;
        T_Buffer<unsigned> cellPointIndices;
        T_Buffer<unsigned> cellTypes;
        T_Buffer<int> cellFaces;
        T_Buffer<double> barycentricFacePlanes;
        T_Buffer<int> cellNeighborCells;
        T_Buffer<int> cellNeighborLocalFaces;
        T_Buffer<int> cellFaceBoundaries;
        T_Buffer<double> cellVolumes;
        T_Buffer<double> lumpedMaterialVertexVolumes;
        T_Buffer<double> cellVolumePrefix;
        T_Buffer<double> sourceStrengthPrefix;
        T_Buffer<double> cellCenters;
        T_Buffer<double> samplePoints;

        unsigned numberOfMaterials;
        unsigned numberOfCells;
        unsigned numberOfPoints;
        unsigned numberOfSamples;
        unsigned numberOfFacesPerCell;
        unsigned numberOfCellVertices;
        unsigned numberOfMeshPoints;
        unsigned numberOfLevels;
        float thickness;
        bool samplePointsAreMeshPoints;
    };

    /**
     * @brief Host preparation arrays with the same layout as TraceView.
     *
     * This type is the single flattening of primitive topology for execution;
     * it is not a compatibility representation. makeResident establishes the
     * explicit host-to-device ownership boundary.
     */
    class TraceData
    {
    public:
        std::vector<double> points;
        std::vector<double> betaVolume;
        std::vector<unsigned> cellMaterialIds;
        std::vector<std::uint8_t> materialActive;
        std::vector<double> materialRefractiveIndices;
        std::vector<double> materialActiveIonDensities;
        std::vector<double> materialFluorescenceLifetimes;
        std::vector<double> materialBulkAttenuations;
        std::vector<double> materialPeakAbsorption;
        std::vector<double> materialPeakEmission;
        std::vector<unsigned> materialCrossSectionOffsets;
        std::vector<double> crossSectionWavelengths;
        std::vector<double> crossSectionAbsorption;
        std::vector<double> crossSectionEmission;
        std::vector<float> surfaceReflectivities;
        std::vector<float> surfaceRefractiveIndexInside;
        std::vector<float> surfaceRefractiveIndexOutside;
        std::vector<unsigned> cellPointIndices;
        std::vector<unsigned> cellTypes;
        std::vector<int> cellFaces;
        std::vector<double> barycentricFacePlanes;
        std::vector<int> cellNeighborCells;
        std::vector<int> cellNeighborLocalFaces;
        std::vector<int> cellFaceBoundaries;
        std::vector<double> cellVolumes;
        std::vector<double> lumpedMaterialVertexVolumes;
        std::vector<double> cellVolumePrefix;
        std::vector<double> sourceStrengthPrefix;
        std::vector<double> cellCenters;
        std::vector<double> samplePoints;
        unsigned numberOfMaterials = 0u;
        unsigned numberOfCells = 0u;
        unsigned numberOfMeshPoints = 0u;
        unsigned numberOfPoints = 0u;
        unsigned numberOfSamples = 0u;
        unsigned numberOfLevels = 1u;
        float thickness = 0.0f;
        unsigned numberOfFacesPerCell = tet4FaceCount;
        unsigned numberOfCellVertices = tet4VertexCount;
        bool samplePointsAreMeshPoints = false;

        TraceData() = default;

        TraceData(
            std::vector<unsigned> cellPointIndices,
            std::vector<unsigned> cellTypes,
            std::vector<int> cellFaces,
            std::vector<int> cellNeighborCells,
            std::vector<int> cellNeighborLocalFaces,
            std::vector<int> cellFaceBoundaries,
            std::vector<double> cellVolumes,
            std::vector<double> points,
            std::vector<double> samplePoints,
            std::vector<double> cellCenters,
            std::vector<double> betaVolume,
            std::vector<unsigned> cellMaterialIds,
            std::vector<std::uint8_t> materialActive,
            std::vector<double> materialRefractiveIndices,
            std::vector<double> materialActiveIonDensities,
            std::vector<double> materialFluorescenceLifetimes,
            std::vector<double> materialBulkAttenuations,
            std::vector<double> materialPeakAbsorption,
            std::vector<double> materialPeakEmission,
            std::vector<unsigned> materialCrossSectionOffsets,
            std::vector<double> crossSectionWavelengths,
            std::vector<double> crossSectionAbsorption,
            std::vector<double> crossSectionEmission,
            std::vector<float> surfaceReflectivities,
            std::vector<float> surfaceRefractiveIndexInside,
            std::vector<float> surfaceRefractiveIndexOutside,
            unsigned structuredNumberOfPoints = 0u,
            unsigned structuredNumberOfLevels = 1u,
            float structuredThickness = 0.0f,
            bool samplePointsAreMeshPoints = false)
            : points(std::move(points))
            , betaVolume(std::move(betaVolume))
            , cellMaterialIds(std::move(cellMaterialIds))
            , materialActive(std::move(materialActive))
            , materialRefractiveIndices(std::move(materialRefractiveIndices))
            , materialActiveIonDensities(std::move(materialActiveIonDensities))
            , materialFluorescenceLifetimes(std::move(materialFluorescenceLifetimes))
            , materialBulkAttenuations(std::move(materialBulkAttenuations))
            , materialPeakAbsorption(std::move(materialPeakAbsorption))
            , materialPeakEmission(std::move(materialPeakEmission))
            , materialCrossSectionOffsets(std::move(materialCrossSectionOffsets))
            , crossSectionWavelengths(std::move(crossSectionWavelengths))
            , crossSectionAbsorption(std::move(crossSectionAbsorption))
            , crossSectionEmission(std::move(crossSectionEmission))
            , surfaceReflectivities(std::move(surfaceReflectivities))
            , surfaceRefractiveIndexInside(std::move(surfaceRefractiveIndexInside))
            , surfaceRefractiveIndexOutside(std::move(surfaceRefractiveIndexOutside))
            , cellPointIndices(std::move(cellPointIndices))
            , cellTypes(std::move(cellTypes))
            , cellFaces(std::move(cellFaces))
            , cellNeighborCells(std::move(cellNeighborCells))
            , cellNeighborLocalFaces(std::move(cellNeighborLocalFaces))
            , cellFaceBoundaries(std::move(cellFaceBoundaries))
            , cellVolumes(std::move(cellVolumes))
            , cellCenters(std::move(cellCenters))
            , samplePoints(std::move(samplePoints))
            , numberOfMaterials(static_cast<unsigned>(this->materialActive.size()))
            , numberOfCells(static_cast<unsigned>(this->cellTypes.size()))
            , numberOfMeshPoints(static_cast<unsigned>(this->points.size() / 3u))
            , numberOfPoints(
                  structuredNumberOfPoints == 0u ? static_cast<unsigned>(this->samplePoints.size() / 3u)
                                                 : structuredNumberOfPoints)
            , numberOfSamples(static_cast<unsigned>(this->samplePoints.size() / 3u))
            , numberOfLevels(structuredNumberOfLevels == 0u ? 1u : structuredNumberOfLevels)
            , thickness(structuredThickness)
            , samplePointsAreMeshPoints(samplePointsAreMeshPoints)
        {
            rebuildStaticPrefixes();
            precomputeBarycentricFacePlanes();
        }

        /** @brief Recompute cell-volume, lumped-vertex-volume, and source-strength lookup arrays. */
        void rebuildStaticPrefixes()
        {
            cellVolumePrefix.resize(cellVolumes.size());
            std::partial_sum(cellVolumes.begin(), cellVolumes.end(), cellVolumePrefix.begin());

            std::size_t const expectedCellPointCount = cellVolumes.size() * numberOfCellVertices;
            bool const hasCompleteCellTopology
                = cellPointIndices.size() == expectedCellPointCount && cellMaterialIds.size() >= cellVolumes.size();
            if(hasCompleteCellTopology)
            {
                lumpedMaterialVertexVolumes.assign(
                    static_cast<std::size_t>(numberOfMaterials) * numberOfMeshPoints,
                    0.0);
                for(std::size_t cell = 0u; cell < cellVolumes.size(); ++cell)
                {
                    double const share = cellVolumes[cell] / static_cast<double>(numberOfCellVertices);
                    for(std::size_t localVertex = 0u; localVertex < numberOfCellVertices; ++localVertex)
                    {
                        std::size_t const point = cellPointIndices[cell * numberOfCellVertices + localVertex];
                        std::size_t const materialVertex
                            = static_cast<std::size_t>(cellMaterialIds[cell]) * numberOfMeshPoints + point;
                        lumpedMaterialVertexVolumes.at(materialVertex) += share;
                    }
                }
            }
            else
                lumpedMaterialVertexVolumes.clear();

            rebuildSourceStrengthPrefix();
        }

        /** @brief Recompute the cumulative active source strength in cell order. */
        void rebuildSourceStrengthPrefix()
        {
            sourceStrengthPrefix.resize(cellVolumes.size());
            double runningStrength = 0.0;
            for(std::size_t cell = 0u; cell < cellVolumes.size(); ++cell)
            {
                double const beta = cell < betaVolume.size() ? betaVolume[cell] : 0.0;
                unsigned const material = cellMaterialIds.at(cell);
                double const density = materialActiveIonDensities.at(material);
                double const lifetime = materialFluorescenceLifetimes.at(material);
                if(materialActive.at(material) != 0u && lifetime > 0.0)
                    runningStrength += beta * cellVolumes[cell] * density / lifetime;
                sourceStrengthPrefix[cell] = runningStrength;
            }
        }

        /**
         * @param values Cell-ordered excited-state fractions to adopt.
         * @post The cumulative source-strength prefix reflects the new values.
         */
        void setBetaVolume(std::vector<double> values)
        {
            betaVolume = std::move(values);
            rebuildSourceStrengthPrefix();
        }

        /**
         * @brief Compare all arrays that are refreshed as material-resident data.
         * @param other Prepared trace to compare against this trace.
         * @return Whether material assignment, coefficients, and spectra are equal.
         */
        [[nodiscard]] bool hasSameMaterialData(TraceData const& other) const
        {
            return numberOfMaterials == other.numberOfMaterials && cellMaterialIds == other.cellMaterialIds
                   && materialActive == other.materialActive
                   && materialRefractiveIndices == other.materialRefractiveIndices
                   && materialActiveIonDensities == other.materialActiveIonDensities
                   && materialFluorescenceLifetimes == other.materialFluorescenceLifetimes
                   && materialBulkAttenuations == other.materialBulkAttenuations
                   && materialPeakAbsorption == other.materialPeakAbsorption
                   && materialPeakEmission == other.materialPeakEmission
                   && materialCrossSectionOffsets == other.materialCrossSectionOffsets
                   && crossSectionWavelengths == other.crossSectionWavelengths
                   && crossSectionAbsorption == other.crossSectionAbsorption
                   && crossSectionEmission == other.crossSectionEmission;
        }

        /**
         * @brief Adopt material arrays from an explicitly prepared update.
         *
         * Cell ordering and material assignment are immutable during a run;
         * only material coefficients and resizable spectra may change.
         * @param other Prepared trace supplying replacement material arrays.
         * @throws std::runtime_error If cell or material layout differs.
         */
        void replaceMaterialData(TraceData const& other)
        {
            if(numberOfCells != other.numberOfCells || numberOfMaterials != other.numberOfMaterials
               || cellMaterialIds != other.cellMaterialIds)
                throw std::runtime_error("dynamic material update changed the tracing-domain layout");
            materialActive = other.materialActive;
            materialRefractiveIndices = other.materialRefractiveIndices;
            materialActiveIonDensities = other.materialActiveIonDensities;
            materialFluorescenceLifetimes = other.materialFluorescenceLifetimes;
            materialBulkAttenuations = other.materialBulkAttenuations;
            materialPeakAbsorption = other.materialPeakAbsorption;
            materialPeakEmission = other.materialPeakEmission;
            materialCrossSectionOffsets = other.materialCrossSectionOffsets;
            crossSectionWavelengths = other.crossSectionWavelengths;
            crossSectionAbsorption = other.crossSectionAbsorption;
            crossSectionEmission = other.crossSectionEmission;
            rebuildSourceStrengthPrefix();
        }

        /** @brief Rebuild affine opposite-vertex coordinates for every valid tetrahedral face. */
        void precomputeBarycentricFacePlanes()
        {
            barycentricFacePlanes.assign(
                static_cast<std::size_t>(numberOfCells) * numberOfFacesPerCell * tet4BarycentricPlaneWidth,
                0.0);
            for(unsigned cell = 0u; cell < numberOfCells; ++cell)
            {
                for(unsigned localFace = 0u; localFace < numberOfFacesPerCell; ++localFace)
                {
                    unsigned const faceOffset = (cell * numberOfFacesPerCell + localFace) * tet4FaceWidth;
                    if(faceOffset + tet4FaceWidth > cellFaces.size())
                    {
                        continue;
                    }
                    int const p0 = cellFaces[faceOffset];
                    int const p1 = cellFaces[faceOffset + 1u];
                    int const p2 = cellFaces[faceOffset + 2u];
                    if(p0 < 0 || p1 < 0 || p2 < 0 || static_cast<unsigned>(p0) >= numberOfMeshPoints
                       || static_cast<unsigned>(p1) >= numberOfMeshPoints
                       || static_cast<unsigned>(p2) >= numberOfMeshPoints)
                    {
                        continue;
                    }

                    int opposite = -1;
                    unsigned const cellOffset = cell * numberOfCellVertices;
                    for(unsigned localVertex = 0u; localVertex < numberOfCellVertices; ++localVertex)
                    {
                        if(cellOffset + localVertex >= cellPointIndices.size())
                        {
                            break;
                        }
                        unsigned const vertex = cellPointIndices[cellOffset + localVertex];
                        if(vertex != static_cast<unsigned>(p0) && vertex != static_cast<unsigned>(p1)
                           && vertex != static_cast<unsigned>(p2))
                        {
                            opposite = static_cast<int>(vertex);
                            break;
                        }
                    }
                    if(opposite < 0 || static_cast<unsigned>(opposite) >= numberOfMeshPoints)
                    {
                        continue;
                    }

                    auto const point = [this](unsigned const index)
                    {
                        return Point{
                            points[index],
                            points[index + numberOfMeshPoints],
                            points[index + 2u * numberOfMeshPoints]};
                    };
                    Point const a = point(static_cast<unsigned>(p0));
                    Point const b = point(static_cast<unsigned>(p1));
                    Point const c = point(static_cast<unsigned>(p2));
                    Point const oppositePoint = point(static_cast<unsigned>(opposite));
                    Point const normal = cross(b - a, c - a);
                    double const denominator = dot(normal, oppositePoint - a);
                    if(std::abs(denominator) <= std::numeric_limits<double>::epsilon())
                    {
                        continue;
                    }
                    Point const gradient = normal * (1.0 / denominator);
                    unsigned const planeOffset = (cell * numberOfFacesPerCell + localFace) * tet4BarycentricPlaneWidth;
                    barycentricFacePlanes[planeOffset] = gradient.x;
                    barycentricFacePlanes[planeOffset + 1u] = gradient.y;
                    barycentricFacePlanes[planeOffset + 2u] = gradient.z;
                    barycentricFacePlanes[planeOffset + 3u] = -dot(gradient, a);
                }
            }
        }

        /**
         * @brief Bind all host arrays to allocations on one device.
         * @tparam T_Device Alpaka host-side device type.
         * @param device Device that owns the returned allocations.
         * @return Resident trace whose host views refer to this object's vectors.
         * @note This prepares allocations but does not upload them; call `ResidentTrace::toDevice`.
         */
        template<alpaka::onHost::concepts::Device T_Device>
        [[nodiscard]] ResidentTrace<T_Device> makeResident(T_Device& device)
        {
            rebuildStaticPrefixes();
            return ResidentTrace<T_Device>{
                device,
                numberOfMaterials,
                numberOfCells,
                numberOfPoints,
                numberOfSamples,
                numberOfFacesPerCell,
                numberOfCellVertices,
                numberOfMeshPoints,
                numberOfLevels,
                thickness,
                samplePointsAreMeshPoints,
                points,
                betaVolume,
                cellMaterialIds,
                materialActive,
                materialRefractiveIndices,
                materialActiveIonDensities,
                materialFluorescenceLifetimes,
                materialBulkAttenuations,
                materialPeakAbsorption,
                materialPeakEmission,
                materialCrossSectionOffsets,
                crossSectionWavelengths,
                crossSectionAbsorption,
                crossSectionEmission,
                surfaceReflectivities,
                surfaceRefractiveIndexInside,
                surfaceRefractiveIndexOutside,
                cellPointIndices,
                cellTypes,
                cellFaces,
                barycentricFacePlanes,
                cellNeighborCells,
                cellNeighborLocalFaces,
                cellFaceBoundaries,
                cellVolumes,
                lumpedMaterialVertexVolumes,
                cellVolumePrefix,
                sourceStrengthPrefix,
                cellCenters,
                samplePoints};
        }
    };

    template<alpaka::onHost::concepts::Device T_Device>
    void ResidentTrace<T_Device>::refreshMaterials(TraceData& trace, concepts::Queue auto const& queue)
    {
        if(numberOfMaterials != trace.numberOfMaterials || numberOfCells != trace.numberOfCells)
            throw std::runtime_error("material refresh changed the tracing-domain layout");
        materialActive = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialActive);
        materialRefractiveIndices = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialRefractiveIndices);
        materialActiveIonDensities = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialActiveIonDensities);
        materialFluorescenceLifetimes
            = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialFluorescenceLifetimes);
        materialBulkAttenuations = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialBulkAttenuations);
        materialPeakAbsorption = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialPeakAbsorption);
        materialPeakEmission = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialPeakEmission);
        materialCrossSectionOffsets = hase::alpakaUtils::getHybridBuffer(m_device, trace.materialCrossSectionOffsets);
        crossSectionWavelengths = hase::alpakaUtils::getHybridBuffer(m_device, trace.crossSectionWavelengths);
        crossSectionAbsorption = hase::alpakaUtils::getHybridBuffer(m_device, trace.crossSectionAbsorption);
        crossSectionEmission = hase::alpakaUtils::getHybridBuffer(m_device, trace.crossSectionEmission);
        materialActive.toDevice(queue);
        materialRefractiveIndices.toDevice(queue);
        materialActiveIonDensities.toDevice(queue);
        materialFluorescenceLifetimes.toDevice(queue);
        materialBulkAttenuations.toDevice(queue);
        materialPeakAbsorption.toDevice(queue);
        materialPeakEmission.toDevice(queue);
        materialCrossSectionOffsets.toDevice(queue);
        crossSectionWavelengths.toDevice(queue);
        crossSectionAbsorption.toDevice(queue);
        crossSectionEmission.toDevice(queue);
    }

} // namespace hase::data
