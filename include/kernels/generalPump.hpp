/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/memory.hpp>
#include <alpakaUtils/utils.hpp>
#include <concepts/concepts.hpp>
#include <core/SimulationControls.hpp>
#include <core/compileTimeConfig.hpp>
#include <core/geometry.hpp>
#include <core/hostRoutineTiming.hpp>
#include <core/physicalConstants.hpp>
#include <core/surfaceReservoir.hpp>
#include <data/TraceData.hpp>
#include <kernels/forward/barycentric.hpp>
#include <kernels/forward/policyRay.hpp>
#include <kernels/forward/rayTransition.hpp>
#include <kernels/forward/rayWalk.hpp>
#include <kernels/vertexAccumulation.hpp>
#include <random/random.hpp>
#include <random/randomEngine.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace hase::kernels
{
    /** @brief Host representation of one oriented exterior pump face or subregion. */
    struct PumpBoundaryFace
    {
        unsigned cell = 0u;
        unsigned localFace = 0u;
        int domain = 0;
        std::array<hase::core::Point, 3u> vertices;
        hase::core::Point centroid;
        hase::core::Point normal;
        double area = 0.0;
    };

    /** @brief Mutable physical and relay state for one pump-ray history. */
    struct GeneralPumpRayState
        : hase::kernels::forward::ray::TraversalState
        , hase::kernels::forward::ray::SrmPositionStorage<
              typename std::remove_cvref_t<ALPAKA_TYPEOF(hase::kernels::forward::ray::pumpSrmPolicy)>::PositionPolicy>
    {
        double power = 0.0;
        double wavelength = 0.0;
        unsigned relayIndex = 0u;
    };

    static_assert(std::derived_from<GeneralPumpRayState, hase::kernels::forward::ray::BarycentricSrmPositionStorage>);

    /**
     * @param domains Accepted boundary-domain identifiers.
     * @param domain Identifier to find.
     * @return Whether `domain` occurs in `domains`.
     */
    [[nodiscard]] inline bool containsDomain(std::vector<int> const& domains, int const domain)
    {
        return std::find(domains.begin(), domains.end(), domain) != domains.end();
    }

    /**
     * @param mesh Host trace storing point components in structure-of-arrays order.
     * @param point Geometric point index.
     * @return Cartesian point assembled from the host arrays.
     */
    [[nodiscard]] inline hase::core::Point hostPoint(hase::data::TraceData const& mesh, unsigned const point)
    {
        return {
            mesh.points[point],
            mesh.points[point + mesh.numberOfMeshPoints],
            mesh.points[point + 2u * mesh.numberOfMeshPoints]};
    }

    /**
     * @param mesh Host trace containing exterior-face geometry and boundary ids.
     * @param domains Boundary-domain identifiers selected by the injector.
     * @return Matching exterior faces with inward launch geometry.
     */
    [[nodiscard]] inline std::vector<PumpBoundaryFace> pumpBoundaryFaces(
        hase::data::TraceData const& mesh,
        std::vector<int> const& domains)
    {
        std::vector<PumpBoundaryFace> result;
        for(unsigned cell = 0u; cell < mesh.numberOfCells; ++cell)
        {
            for(unsigned face = 0u; face < mesh.numberOfFacesPerCell; ++face)
            {
                unsigned const faceIndex = cell * mesh.numberOfFacesPerCell + face;
                int const domain = mesh.cellFaceBoundaries[faceIndex];
                if(mesh.cellNeighborCells[faceIndex] >= 0 || !containsDomain(domains, domain))
                    continue;
                PumpBoundaryFace info;
                info.cell = cell;
                info.localFace = face;
                info.domain = domain;
                for(unsigned vertex = 0u; vertex < 3u; ++vertex)
                {
                    int const point = mesh.cellFaces[faceIndex * 3u + vertex];
                    if(point < 0)
                        throw std::runtime_error("pump boundary face contains an invalid point");
                    info.vertices[vertex] = hostPoint(mesh, static_cast<unsigned>(point));
                }
                info.centroid = (info.vertices[0] + info.vertices[1] + info.vertices[2]) * (1.0 / 3.0);
                auto normal
                    = hase::core::cross(info.vertices[1] - info.vertices[0], info.vertices[2] - info.vertices[0]);
                double const twiceArea = normal.euclidLength();
                if(twiceArea <= 0.0)
                    continue;
                info.area = 0.5 * twiceArea;
                info.normal = normal * (1.0 / twiceArea);
                core::Point const center{
                    mesh.cellCenters[cell],
                    mesh.cellCenters[cell + mesh.numberOfCells],
                    mesh.cellCenters[cell + 2u * mesh.numberOfCells]};
                if(hase::core::dot(info.normal, center - info.centroid) > 0.0)
                    info.normal = info.normal * -1.0;
                result.push_back(info);
            }
        }
        return result;
    }

    /** @return Unit vector parallel to `value`, or zero for negligible length. */
    [[nodiscard]] inline hase::core::Point hostNormalize(hase::core::Point const value)
    {
        double const length = value.euclidLength();
        if(length <= 0.0)
            return {0.0, 0.0, 0.0};
        return value * (1.0 / length);
    }

    /** @return Deterministic unit vector perpendicular to `normal`. */
    [[nodiscard]] inline hase::core::Point perpendicular(hase::core::Point const normal)
    {
        core::Point reference
            = std::abs(normal.x) < 0.9 ? hase::core::Point{1.0, 0.0, 0.0} : hase::core::Point{0.0, 1.0, 0.0};
        return hostNormalize(hase::core::cross(normal, reference));
    }

    /**
     * @param position Cartesian position on the injector aperture.
     * @param profile Prepared uniform or super-Gaussian profile parameters.
     * @return Non-negative relative spatial source weight.
     */
    [[nodiscard]] inline double pumpProfileWeight(
        hase::core::PumpProfileParameters const& profile,
        hase::core::Point const point)
    {
        if(profile.kind == 0u)
            return 1.0;
        hase::core::Point const relative
            = point - hase::core::Point{profile.center[0], profile.center[1], profile.center[2]};
        double const u
            = hase::core::dot(relative, hase::core::Point{profile.axisU[0], profile.axisU[1], profile.axisU[2]})
              / profile.radiusU;
        double const v
            = hase::core::dot(relative, hase::core::Point{profile.axisV[0], profile.axisV[1], profile.axisV[2]})
              / profile.radiusV;
        return std::exp(-std::pow(std::sqrt(u * u + v * v), profile.exponent));
    }

    /**
     * @param face Triangular injector region.
     * @param profile Prepared spatial profile.
     * @return Quadrature estimate of the profile integral over the region.
     */
    [[nodiscard]] inline double pumpEntryWeight(
        PumpBoundaryFace const& face,
        hase::core::PumpProfileParameters const& profile)
    {
        // Seven-point Dunavant quadrature.  The CDF must represent the
        // aperture's spatial entry distribution, not just its triangle areas:
        // after a region is selected, rejection sampling supplies the matching
        // conditional distribution inside that region.
        constexpr std::array<std::array<double, 3u>, 7u> barycentric{{
            {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
            {0.059715871789770, 0.470142064105115, 0.470142064105115},
            {0.470142064105115, 0.059715871789770, 0.470142064105115},
            {0.470142064105115, 0.470142064105115, 0.059715871789770},
            {0.797426985353087, 0.101286507323456, 0.101286507323456},
            {0.101286507323456, 0.797426985353087, 0.101286507323456},
            {0.101286507323456, 0.101286507323456, 0.797426985353087},
        }};
        constexpr std::array<double, 7u> weights{
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827};

        double integral = 0.0;
        for(std::size_t sample = 0u; sample < barycentric.size(); ++sample)
        {
            auto const& coordinate = barycentric[sample];
            hase::core::Point const point = face.vertices[0] * coordinate[0] + face.vertices[1] * coordinate[1]
                                            + face.vertices[2] * coordinate[2];
            integral += weights[sample] * pumpProfileWeight(profile, point);
        }
        return face.area * integral;
    }

    /**
     * @brief Recursively subdivide one triangular injector face into four-way regions.
     * @param face Region to subdivide while preserving boundary metadata.
     * @param remainingSubdivisions Number of recursive subdivision levels.
     * @param regions Output vector receiving leaf regions in deterministic order.
     */
    inline void appendPumpEntryRegions(
        PumpBoundaryFace const& face,
        unsigned const remainingSubdivisions,
        std::vector<PumpBoundaryFace>& regions)
    {
        if(remainingSubdivisions == 0u)
        {
            regions.push_back(face);
            return;
        }

        auto const midpoint = [](hase::core::Point const a, hase::core::Point const b) { return (a + b) * 0.5; };
        hase::core::Point const midpoint01 = midpoint(face.vertices[0], face.vertices[1]);
        hase::core::Point const midpoint12 = midpoint(face.vertices[1], face.vertices[2]);
        hase::core::Point const midpoint20 = midpoint(face.vertices[2], face.vertices[0]);
        std::array<std::array<hase::core::Point, 3u>, 4u> const vertices{{
            {face.vertices[0], midpoint01, midpoint20},
            {midpoint01, face.vertices[1], midpoint12},
            {midpoint20, midpoint12, face.vertices[2]},
            {midpoint01, midpoint12, midpoint20},
        }};
        for(auto const& regionVertices : vertices)
        {
            PumpBoundaryFace region = face;
            region.vertices = regionVertices;
            region.centroid = (regionVertices[0] + regionVertices[1] + regionVertices[2]) * (1.0 / 3.0);
            region.area = face.area * 0.25;
            appendPumpEntryRegions(region, remainingSubdivisions - 1u, regions);
        }
    }

    /**
     * @param faces Exterior injector faces.
     * @return Deterministically subdivided spatial regions used by systematic sampling.
     */
    [[nodiscard]] inline std::vector<PumpBoundaryFace> pumpEntryRegions(std::vector<PumpBoundaryFace> const& faces)
    {
        // Spatial regions make the systematic CDF cover the continuous entry
        // aperture.  A face-only CDF leaves all within-face variation random,
        // which is especially noisy for a nonuniform beam profile.
        constexpr unsigned subdivisionDepth = 2u;
        constexpr std::size_t regionsPerFace = 1u << (2u * subdivisionDepth);
        std::vector<PumpBoundaryFace> regions;
        regions.reserve(faces.size() * regionsPerFace);
        for(auto const& face : faces)
            appendPumpEntryRegions(face, subdivisionDepth, regions);
        return regions;
    }

    /**
     * @param entryCdf Cumulative non-negative entry-region weights.
     * @param target Target in the CDF's weight domain.
     * @return Selected region index, clamped to the last region.
     */
    [[nodiscard]] inline std::size_t pumpEntryRegionForTarget(std::vector<double> const& entryCdf, double const target)
    {
        if(entryCdf.empty())
            return 0u;
        auto const region = std::upper_bound(entryCdf.cbegin(), entryCdf.cend(), target);
        return region == entryCdf.cend() ? entryCdf.size() - 1u
                                         : static_cast<std::size_t>(std::distance(entryCdf.cbegin(), region));
    }

    /**
     * @param a First triangle vertex.
     * @param b Second triangle vertex.
     * @param c Third triangle vertex.
     * @param rng Random engine advanced by area-uniform sampling.
     * @return Uniformly distributed point on the triangle.
     */
    [[nodiscard]] inline hase::core::Point sampleTriangle(
        PumpBoundaryFace const& face,
        std::uniform_random_bit_generator auto& rng)
    {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double u = uniform(rng);
        double v = uniform(rng);
        if(u + v > 1.0)
        {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        return face.vertices[0] + (face.vertices[1] - face.vertices[0]) * u
               + (face.vertices[2] - face.vertices[0]) * v;
    }

    /**
     * @brief Generate a deterministic partition of one pump's global ray set.
     * @param mesh Host trace containing injector geometry.
     * @param source Prepared pump source and sampling distributions.
     * @param globalRayCount Total histories in the unpartitioned source.
     * @param seed Source RNG seed.
     * @param firstRay First global history assigned to this partition.
     * @param localRayCount Maximum histories returned for this partition.
     * @return Host ray states retaining global stratification and equal-power weights.
     */
    [[nodiscard]] inline std::vector<GeneralPumpRayState> samplePumpSource(
        hase::data::TraceData const& mesh,
        hase::core::PumpSourceParameters const& source,
        unsigned const globalRayCount,
        std::uint32_t const seed,
        unsigned const firstRay = 0u,
        unsigned const localRayCount = std::numeric_limits<unsigned>::max())
    {
        unsigned const selectedRayCount = std::min(localRayCount, globalRayCount - std::min(firstRay, globalRayCount));
        unsigned const selectedRayEnd = std::min(globalRayCount, firstRay + selectedRayCount);
        auto const faces = pumpBoundaryFaces(mesh, source.surfaces);
        if(faces.empty())
            throw std::runtime_error("pump source selected no exterior boundary faces");
        if(selectedRayCount == 0u)
            return {};
        auto const entryRegions = pumpEntryRegions(faces);
        std::vector<double> entryCdf;
        entryCdf.reserve(entryRegions.size());
        double totalEntryWeight = 0.0;
        for(auto const& region : entryRegions)
        {
            totalEntryWeight += pumpEntryWeight(region, source.profile);
            entryCdf.push_back(totalEntryWeight);
        }
        if(!(totalEntryWeight > 0.0) || !std::isfinite(totalEntryWeight))
            throw std::runtime_error("pump spatial profile has no finite weight on the selected exterior faces");
        std::discrete_distribution<std::size_t> spectrumDistribution(
            source.spectralWeights.begin(),
            source.spectralWeights.end());
        std::discrete_distribution<std::size_t> angularDistribution(
            source.angularWeights.begin(),
            source.angularWeights.end());
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double const entryStratificationOffset = hase::random::stratifiedUnitOffset(seed);

        std::vector<GeneralPumpRayState> rays;
        rays.reserve(selectedRayCount);
        for(unsigned ray = 0u; ray < selectedRayEnd; ++ray)
        {
            double const entryTarget = (static_cast<double>(ray) + entryStratificationOffset)
                                       / static_cast<double>(globalRayCount) * totalEntryWeight;
            PumpBoundaryFace const* face = &entryRegions[pumpEntryRegionForTarget(entryCdf, entryTarget)];
            hase::core::Point origin;
            bool accepted = false;
            for(unsigned attempt = 0u; attempt < 100000u; ++attempt)
            {
                origin = sampleTriangle(*face, rng);
                if(uniform(rng) <= pumpProfileWeight(source.profile, origin))
                {
                    accepted = true;
                    break;
                }
            }
            if(!accepted)
                throw std::runtime_error("pump spatial profile rejection sampling did not converge");

            std::size_t const angular = angularDistribution(rng);
            double const theta = source.polarAngles[angular];
            double const phi = source.azimuthalAngles[angular];
            hase::core::Point const inward = face->normal * -1.0;
            hase::core::Point const u = perpendicular(inward);
            hase::core::Point const v = hase::core::cross(inward, u);
            hase::core::Point const direction = hostNormalize(
                inward * std::cos(theta) + u * (std::sin(theta) * std::cos(phi))
                + v * (std::sin(theta) * std::sin(phi)));
            std::size_t const spectrum = spectrumDistribution(rng);

            if(ray < firstRay)
                continue;

            GeneralPumpRayState rayState;
            rayState.position = origin;
            rayState.direction = direction;
            rayState.power = source.totalPower / static_cast<double>(globalRayCount);
            rayState.wavelength = source.wavelengths[spectrum];
            rayState.cell = face->cell;
            rayState.forbiddenFace = static_cast<std::int32_t>(face->localFace);
            rays.push_back(rayState);
        }
        return rays;
    }

    /** @brief Pump boundary policy capturing barycentric exit positions. */
    struct StorePumpSrmBoundary
        : hase::kernels::forward::ray::BoundaryPolicySrm<hase::kernels::forward::ray::srmPosition::Barycentric>
    {
        ALPAKA_FN_ACC hase::kernels::forward::ray::BoundaryResult operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            hase::data::TraceView const& mesh,
            GeneralPumpRayState& ray,
            unsigned const cell,
            unsigned const localFace)
        {
            namespace policyRay = hase::kernels::forward::ray;
            policyRay::captureSrmPosition(this->positionPolicy, mesh, cell, localFace, ray.position, ray);
            return policyRay::BoundaryResult::stop;
        }
    };

    struct StorePumpSrmBoundaryFactory
    {
        ALPAKA_FN_ACC auto operator()(unsigned) const
        {
            return StorePumpSrmBoundary{};
        }
    };

    /** @brief Kernel tracing prepared pump rays and accumulating material-vertex exchange. */
    struct TraceGeneralPump
    {
        template<
            alpaka::concepts::IView<double> T_BetaVolumeView,
            alpaka::concepts::IView<double> T_VertexPumpIntegralView>
        struct CellPolicy : forward::ray::behaviourDimension::Cell
        {
            T_BetaVolumeView betaVolume;
            T_VertexPumpIntegralView vertexPumpIntegral;

            ALPAKA_FN_HOST_ACC constexpr CellPolicy(
                T_BetaVolumeView betaVolumeValue,
                T_VertexPumpIntegralView vertexPumpIntegralValue)
                : betaVolume{betaVolumeValue}
                , vertexPumpIntegral{vertexPumpIntegralValue}
            {
            }

            ALPAKA_FN_ACC bool operator()(
                alpaka::onAcc::concepts::Acc auto const& acc,
                data::TraceView const& mesh,
                GeneralPumpRayState& ray,
                unsigned const tet,
                forward::Tet4FaceIntersection const intersection)
            {
                bool const active = mesh.isActive(tet);
                auto const crossSections = mesh.crossSectionsForCell(tet, ray.wavelength);
                double const gain
                    = (active ? mesh.activeIonDensity(tet)
                                    * (betaVolume[tet] * (crossSections.absorption + crossSections.emission)
                                       - crossSections.absorption)
                              : 0.0)
                      - mesh.bulkAttenuation(tet);
                double const exponent = gain * intersection.length;
                if(!alpaka::math::isfinite(exponent) || exponent > 700.0)
                {
                    ray.power = 0.0;
                    return false;
                }
                double const nextPower = ray.power * alpaka::math::exp(exponent);
                if(active && mesh.activeIonDensity(tet) > 0.0)
                {
                    double const integral
                        = (ray.power - nextPower) * ray.wavelength
                          / (hase::core::physicalConstants::planckConstant
                             * hase::core::physicalConstants::speedOfLight * mesh.activeIonDensity(tet));
                    // Clamping and renormalizing protects positivity and exact integral
                    // conservation against round-off at faces.
                    auto const weights = forward::segmentMidpointBarycentricVertexWeights(
                        mesh,
                        tet,
                        ray.position,
                        ray.direction,
                        intersection.length);
                    for(unsigned localVertex = 0u; localVertex < hase::data::tet4VertexCount; ++localVertex)
                    {
                        unsigned const point = mesh.cellPointIndices[tet * mesh.numberOfCellVertices + localVertex]
                                               + mesh.getMaterialId(tet) * mesh.numberOfMeshPoints;
                        alpaka::onAcc::atomicAdd(acc, &vertexPumpIntegral[point], integral * weights[localVertex]);
                    }
                }
                ray.power = nextPower;
                return ray.power != 0.0;
            }
        };

        template<alpaka::concepts::IView<double> T_GeometryView>
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const mesh,
            alpaka::concepts::IView<double> auto betaVolume,
            hase::core::RayGeometryViewSoA<T_GeometryView> geometry,
            alpaka::concepts::IView<double> auto power,
            alpaka::concepts::IView<double> auto wavelength,
            alpaka::concepts::IView<std::uint32_t> auto cell,
            alpaka::concepts::IView<std::int32_t> auto forbiddenFace,
            std::invocable<unsigned> auto boundaryPolicyFactory,
            alpaka::concepts::IView<double> auto vertexPumpIntegral,
            unsigned const rayCount) const
        {
            for(auto [rayIndex] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{rayCount}))
            {
                namespace ray = forward::ray;
                GeneralPumpRayState rayState;
                rayState.position = geometry.positions.at(rayIndex);
                rayState.direction = geometry.directions.at(rayIndex);
                rayState.cell = cell[rayIndex];
                rayState.forbiddenFace = forbiddenFace[rayIndex];
                rayState.power = power[rayIndex];
                rayState.wavelength = wavelength[rayIndex];
                auto boundaryPolicy = boundaryPolicyFactory(rayIndex);
                static_cast<void>(ray::walk(
                    acc,
                    mesh,
                    rayState,
                    ray::RayWalkBehaviour{CellPolicy{betaVolume, vertexPumpIntegral}, boundaryPolicy}));
            }
        }
    };

    /** @brief Area-weighted coordinate frame for a planar relay aperture. */
    struct RelayFrame
    {
        hase::core::Point origin, u, v, normal;
        std::vector<PumpBoundaryFace> faces;
    };

    /** @brief Trivially-copyable device transform for one planar pump relay. */
    struct PumpRelayDeviceDescriptor
    {
        hase::core::Point exitOrigin, exitU, exitV, exitNormal;
        hase::core::Point entryOrigin, entryU, entryV, entryNormal;
        double cosine = 1.0;
        double sine = 0.0;
        double offsetU = 0.0;
        double offsetV = 0.0;
        double tiltU = 0.0;
        double tiltV = 0.0;
        double magnification = 1.0;
        //! Relay throughput applied before inward reinjection at the entry surface.
        double transmission = 1.0;
        int flipU = 1;
        int flipV = 1;
        unsigned entryFaceBegin = 0u;
        unsigned entryFaceEnd = 0u;
    };

    namespace pumpBoundaryPolicy
    {
        struct Relay
        {
        };

        struct SrmBarycentric
        {
        };
    } // namespace pumpBoundaryPolicy

    inline constexpr pumpBoundaryPolicy::Relay pumpRelayPolicy{};
    inline constexpr pumpBoundaryPolicy::SrmBarycentric pumpSrmBarycentricPolicy{};
    inline constexpr alpaka::concepts::CVector auto pumpReservoirSlots
        = hase::core::compileTimeConfig::pumpReservoirSlots;

    template<
        alpaka::concepts::IView<PumpRelayDeviceDescriptor> T_DescriptorView,
        alpaka::concepts::IView<std::uint32_t> T_IndexView,
        alpaka::concepts::IView<double> T_BarycentricView>
    struct DevicePumpRelayBoundary
        : hase::kernels::forward::ray::BoundaryPolicySrm<hase::kernels::forward::ray::srmPosition::Barycentric>
    {
        T_DescriptorView descriptors;
        T_IndexView exitMask;
        T_IndexView entryFaceIds;
        T_IndexView cacheState;
        T_IndexView cacheTargetFace;
        T_BarycentricView cacheBarycentric0;
        T_BarycentricView cacheBarycentric1;
        T_BarycentricView cacheBarycentric2;
        unsigned faceCount;
        unsigned relayCount;
        unsigned rayCount;
        unsigned rayIndex;

        ALPAKA_FN_HOST_ACC constexpr DevicePumpRelayBoundary(
            T_DescriptorView descriptorsValue,
            T_IndexView exitMaskValue,
            T_IndexView entryFaceIdsValue,
            T_IndexView cacheStateValue,
            T_IndexView cacheTargetFaceValue,
            T_BarycentricView cacheBarycentric0Value,
            T_BarycentricView cacheBarycentric1Value,
            T_BarycentricView cacheBarycentric2Value,
            unsigned const faceCountValue,
            unsigned const relayCountValue,
            unsigned const rayCountValue,
            unsigned const rayIndexValue)
            : descriptors{descriptorsValue}
            , exitMask{exitMaskValue}
            , entryFaceIds{entryFaceIdsValue}
            , cacheState{cacheStateValue}
            , cacheTargetFace{cacheTargetFaceValue}
            , cacheBarycentric0{cacheBarycentric0Value}
            , cacheBarycentric1{cacheBarycentric1Value}
            , cacheBarycentric2{cacheBarycentric2Value}
            , faceCount{faceCountValue}
            , relayCount{relayCountValue}
            , rayCount{rayCountValue}
            , rayIndex{rayIndexValue}
        {
        }

        ALPAKA_FN_ACC hase::kernels::forward::ray::BoundaryResult operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            hase::data::TraceView const& mesh,
            GeneralPumpRayState& ray,
            unsigned const cell,
            unsigned const localFace)
        {
            namespace policyRay = hase::kernels::forward::ray;
            if(ray.relayIndex >= relayCount)
            {
                policyRay::captureSrmPosition(this->positionPolicy, mesh, cell, localFace, ray.position, ray);
                return policyRay::BoundaryResult::stop;
            }

            unsigned const relayIndex = ray.relayIndex;
            unsigned const faceId = cell * mesh.numberOfFacesPerCell + localFace;
            if(exitMask[relayIndex * faceCount + faceId] == 0u)
            {
                ray.power = 0.0;
                return policyRay::BoundaryResult::stop;
            }

            auto const& descriptor = descriptors[relayIndex];
            policyRay::captureSrmPosition(this->positionPolicy, mesh, cell, localFace, ray.position, ray);
            core::Point const exitPosition
                = policyRay::restoreSrmPosition(this->positionPolicy, mesh, cell, localFace, ray);
            core::Point const relative = exitPosition - descriptor.exitOrigin;
            double u = core::dot(relative, descriptor.exitU) * static_cast<double>(descriptor.flipU);
            double v = core::dot(relative, descriptor.exitV) * static_cast<double>(descriptor.flipV);
            u *= descriptor.magnification;
            v *= descriptor.magnification;
            double const mappedU = descriptor.cosine * u - descriptor.sine * v + descriptor.offsetU;
            double const mappedV = descriptor.sine * u + descriptor.cosine * v + descriptor.offsetV;
            hase::core::Point mappedPosition
                = descriptor.entryOrigin + descriptor.entryU * mappedU + descriptor.entryV * mappedV;

            unsigned const cacheIndex = relayIndex * rayCount + rayIndex;
            unsigned targetFaceId = cacheTargetFace[cacheIndex];
            policyRay::TriangleBarycentric targetCoordinates{
                cacheBarycentric0[cacheIndex],
                cacheBarycentric1[cacheIndex],
                cacheBarycentric2[cacheIndex]};
            unsigned const cachedState = cacheState[cacheIndex];
            if(cachedState == 0u)
            {
                targetFaceId = faceCount;
                for(unsigned entry = descriptor.entryFaceBegin; entry < descriptor.entryFaceEnd; ++entry)
                {
                    unsigned const candidateFaceId = entryFaceIds[entry];
                    unsigned const candidateCell = candidateFaceId / mesh.numberOfFacesPerCell;
                    unsigned const candidateLocalFace = candidateFaceId % mesh.numberOfFacesPerCell;
                    auto const coordinates = policyRay::triangleBarycentricCoordinates(
                        mesh,
                        candidateCell,
                        candidateLocalFace,
                        mappedPosition);
                    constexpr double tolerance = 1.0e-10;
                    if(coordinates[0u] >= -tolerance && coordinates[1u] >= -tolerance && coordinates[2u] >= -tolerance)
                    {
                        targetFaceId = candidateFaceId;
                        targetCoordinates = coordinates;
                        break;
                    }
                }
                cacheTargetFace[cacheIndex] = targetFaceId;
                cacheBarycentric0[cacheIndex] = targetCoordinates[0u];
                cacheBarycentric1[cacheIndex] = targetCoordinates[1u];
                cacheBarycentric2[cacheIndex] = targetCoordinates[2u];
                cacheState[cacheIndex] = targetFaceId < faceCount ? 1u : 2u;
            }
            else if(cachedState == 2u)
            {
                ray.power = 0.0;
                return policyRay::BoundaryResult::stop;
            }

            if(targetFaceId >= faceCount)
            {
                ray.power = 0.0;
                return policyRay::BoundaryResult::stop;
            }

            unsigned const targetCell = targetFaceId / mesh.numberOfFacesPerCell;
            unsigned const targetLocalFace = targetFaceId % mesh.numberOfFacesPerCell;
            mappedPosition
                = policyRay::positionFromTriangleBarycentric(mesh, targetCell, targetLocalFace, targetCoordinates);
            hase::core::Point const oldDirection = ray.direction;
            double const du = hase::core::dot(oldDirection, descriptor.exitU) * static_cast<double>(descriptor.flipU);
            double const dv = hase::core::dot(oldDirection, descriptor.exitV) * static_cast<double>(descriptor.flipV);
            double const mappedDu = descriptor.cosine * du - descriptor.sine * dv + descriptor.tiltU;
            double const mappedDv = descriptor.sine * du + descriptor.cosine * dv + descriptor.tiltV;
            double const normalMagnitude = alpaka::math::abs(hase::core::dot(oldDirection, descriptor.exitNormal));
            ray.direction = hase::kernels::forward::normalize(
                descriptor.entryU * mappedDu + descriptor.entryV * mappedDv
                - descriptor.entryNormal * normalMagnitude);
            ray.position = mappedPosition;
            ray.cell = targetCell;
            ray.forbiddenFace = static_cast<std::int32_t>(targetLocalFace);
            ray.boundaryBarycentric = targetCoordinates;
            ray.power *= descriptor.transmission;
            ++ray.relayIndex;
            return ray.power == 0.0 ? policyRay::BoundaryResult::stop : policyRay::BoundaryResult::continueTraversal;
        }
    };

    template<
        alpaka::concepts::IView<PumpRelayDeviceDescriptor> T_DescriptorView,
        alpaka::concepts::IView<std::uint32_t> T_IndexView,
        alpaka::concepts::IView<double> T_BarycentricView>
    struct DevicePumpRelayBoundaryFactory
    {
        T_DescriptorView descriptors;
        T_IndexView exitMask;
        T_IndexView entryFaceIds;
        T_IndexView cacheState;
        T_IndexView cacheTargetFace;
        T_BarycentricView cacheBarycentric0;
        T_BarycentricView cacheBarycentric1;
        T_BarycentricView cacheBarycentric2;
        unsigned faceCount;
        unsigned relayCount;
        unsigned rayCount;

        ALPAKA_FN_ACC auto operator()(unsigned const rayIndex) const
        {
            return DevicePumpRelayBoundary{
                descriptors,
                exitMask,
                entryFaceIds,
                cacheState,
                cacheTargetFace,
                cacheBarycentric0,
                cacheBarycentric1,
                cacheBarycentric2,
                faceCount,
                relayCount,
                rayCount,
                rayIndex};
        }
    };

    template<
        alpaka::concepts::IView<PumpRelayDeviceDescriptor> T_DescriptorView,
        alpaka::concepts::IView<std::uint32_t> T_SurfaceMaskView,
        alpaka::concepts::SpecializationOf<hase::kernels::forward::SurfaceReservoirSpans> T_Reservoir,
        alpaka::rand::concepts::UniformRandomEngine T_Rng>
    struct PumpSurfaceReservoirBoundary
        : hase::kernels::forward::ray::BoundaryPolicySrm<hase::kernels::forward::ray::srmPosition::Barycentric>
    {
        T_DescriptorView descriptors;
        T_SurfaceMaskView surfaceMask;
        T_Reservoir reservoir;
        T_Rng rng;
        std::uint32_t faceCount;
        std::uint32_t reflectionIndex;
        std::uint32_t candidateIndex;
        bool captureOutput;

        ALPAKA_FN_HOST_ACC constexpr PumpSurfaceReservoirBoundary(
            T_DescriptorView descriptorsValue,
            T_SurfaceMaskView surfaceMaskValue,
            T_Reservoir reservoirValue,
            T_Rng rngValue,
            std::uint32_t const faceCountValue,
            std::uint32_t const reflectionIndexValue,
            std::uint32_t const candidateIndexValue,
            bool const captureOutputValue)
            : descriptors{descriptorsValue}
            , surfaceMask{surfaceMaskValue}
            , reservoir{reservoirValue}
            , rng{rngValue}
            , faceCount{faceCountValue}
            , reflectionIndex{reflectionIndexValue}
            , candidateIndex{candidateIndexValue}
            , captureOutput{captureOutputValue}
        {
        }

        ALPAKA_FN_ACC hase::kernels::forward::ray::BoundaryResult operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const& mesh,
            GeneralPumpRayState& ray,
            unsigned const cell,
            unsigned const localFace)
        {
            namespace policyRay = hase::kernels::forward::ray;
            if(!captureOutput)
                return policyRay::BoundaryResult::stop;

            std::uint32_t const faceId = cell * mesh.numberOfFacesPerCell + localFace;
            if(surfaceMask[reflectionIndex * faceCount + faceId] == 0u)
                return policyRay::BoundaryResult::stop;

            hase::core::Direction const direction = hase::kernels::forward::reflectedDirection(
                ray.direction,
                hase::kernels::forward::outwardFaceNormal(mesh, cell, localFace));
            return hase::kernels::forward::storeSurfaceReservoirBoundarySample(
                acc,
                mesh,
                ray,
                cell,
                localFace,
                hase::kernels::forward::SurfaceReservoirBoundarySample{
                    direction,
                    ray.power * descriptors[reflectionIndex].transmission,
                    ray.wavelength},
                reservoir,
                candidateIndex,
                rng);
        }
    };

    template<
        alpaka::concepts::IView<PumpRelayDeviceDescriptor> T_DescriptorView,
        alpaka::concepts::IView<std::uint32_t> T_SurfaceMaskView,
        alpaka::concepts::SpecializationOf<hase::kernels::forward::SurfaceReservoirSpans> T_Reservoir>
    struct PumpSurfaceReservoirBoundaryFactory
    {
        T_DescriptorView descriptors;
        T_SurfaceMaskView surfaceMask;
        T_Reservoir reservoir;
        std::uint32_t faceCount;
        std::uint32_t reflectionIndex;
        std::uint32_t rngSeed;
        std::uint32_t pass;
        bool captureOutput;

        ALPAKA_FN_ACC auto operator()(unsigned const rayIndex) const
        {
            return PumpSurfaceReservoirBoundary{
                descriptors,
                surfaceMask,
                reservoir,
                hase::random::makeRandomEngine(rngSeed, (static_cast<std::uint64_t>(pass) << 32u) | rayIndex),
                faceCount,
                reflectionIndex,
                rayIndex,
                captureOutput};
        }
    };

    struct TracePumpSurfaceReservoir
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const mesh,
            alpaka::concepts::IView<double> auto betaVolume,
            alpaka::concepts::SpecializationOf<hase::kernels::forward::SurfaceReservoirSpans> auto input,
            alpaka::concepts::SpecializationOf<hase::kernels::forward::SurfaceReservoirSamplingCdfSpans> auto sampling,
            alpaka::concepts::SpecializationOf<hase::kernels::forward::SurfaceReservoirSpans> auto output,
            alpaka::concepts::IView<PumpRelayDeviceDescriptor> auto descriptors,
            alpaka::concepts::IView<std::uint32_t> auto surfaceMask,
            alpaka::concepts::IView<double> auto vertexPumpIntegral,
            std::uint32_t const faceCount,
            std::uint32_t const rayCount,
            std::uint32_t const reflectionIndex,
            double const sourcePower,
            std::uint32_t const rngSeed,
            std::uint32_t const pass,
            bool const captureOutput) const
        {
            for(auto [rayIndex] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{rayCount}))
            {
                auto rng
                    = hase::random::makeRandomEngine(rngSeed, (static_cast<std::uint64_t>(pass) << 32u) | rayIndex);
                hase::kernels::forward::SurfaceReservoirSample const sample
                    = hase::kernels::forward::sampleSurfaceReservoir(input, sampling, faceCount, rayIndex, rng);
                if(!sample.valid)
                    continue;

                std::uint32_t const cell = sample.faceId / mesh.numberOfFacesPerCell;
                std::uint32_t const localFace = sample.faceId % mesh.numberOfFacesPerCell;
                GeneralPumpRayState ray;
                ray.position = hase::kernels::forward::restoreSurfaceReservoirPosition(
                    input.positionSpans,
                    mesh,
                    cell,
                    localFace,
                    sample.slotIndex);
                ray.direction = hase::kernels::forward::normalize(input.directions.at(sample.slotIndex));
                ray.power = sourcePower;
                ray.wavelength = input.wavelengths[sample.slotIndex];
                ray.cell = cell;
                ray.forbiddenFace = static_cast<std::int32_t>(localFace);
                static_cast<void>(hase::kernels::forward::ray::walk(
                    acc,
                    mesh,
                    ray,
                    hase::kernels::forward::ray::RayWalkBehaviour{
                        TraceGeneralPump::CellPolicy{betaVolume, vertexPumpIntegral},
                        PumpSurfaceReservoirBoundary{
                            descriptors,
                            surfaceMask,
                            output,
                            rng,
                            faceCount,
                            reflectionIndex,
                            rayIndex,
                            captureOutput}}));
            }
        }
    };

    /**
     * @param mesh Host trace containing selected boundary faces.
     * @param domains Boundary-domain identifiers forming one relay aperture.
     * @return Area-weighted origin and orthonormal aperture frame.
     * @throws std::runtime_error If the selection is empty or geometrically invalid.
     */
    [[nodiscard]] inline RelayFrame makeRelayFrame(hase::data::TraceData const& mesh, std::vector<int> const& domains)
    {
        RelayFrame frame{};
        frame.faces = pumpBoundaryFaces(mesh, domains);
        if(frame.faces.empty())
            throw std::runtime_error("pump relay selected no exterior faces");
        double totalArea = 0.0;
        for(auto const& face : frame.faces)
        {
            frame.origin = frame.origin + face.centroid * face.area;
            frame.normal = frame.normal + face.normal * face.area;
            totalArea += face.area;
        }
        frame.origin = frame.origin * (1.0 / totalArea);
        frame.normal = hostNormalize(frame.normal);
        frame.u = hostNormalize(
            (frame.faces.front().vertices[1] - frame.faces.front().vertices[0])
            - frame.normal
                  * hase::core::dot(frame.faces.front().vertices[1] - frame.faces.front().vertices[0], frame.normal));
        if(frame.u.euclidLength() == 0.0)
            frame.u = perpendicular(frame.normal);
        frame.v = hase::core::cross(frame.normal, frame.u);
        double scale = 0.0;
        for(auto const& face : frame.faces)
            for(auto const& vertex : face.vertices)
                scale = std::max(scale, (vertex - frame.origin).euclidLength());
        for(auto const& face : frame.faces)
        {
            constexpr double minimumGeometryScale = 1.0e-2;
            if(std::abs(hase::core::dot(face.centroid - frame.origin, frame.normal))
               > 1.0e-8 * std::max(minimumGeometryScale, scale))
                throw std::runtime_error("pump relay surfaces must be coplanar");
        }
        return frame;
    }

    /** @brief Device-ready relay descriptors, masks, and entry-face indices. */
    struct PumpRelayGeometry
    {
        std::vector<PumpRelayDeviceDescriptor> descriptors;
        std::vector<unsigned> exitMask;
        std::vector<unsigned> entryFaceIds;
        unsigned faceCount = 0u;
    };

    /**
     * @param mesh Host trace containing boundary geometry.
     * @param relays Prepared planar relay controls.
     * @return Flattened relay geometry for device upload.
     */
    [[nodiscard]] inline PumpRelayGeometry preparePumpRelayGeometry(
        hase::data::TraceData const& mesh,
        std::vector<hase::core::PumpRelayParameters> const& relays)
    {
        PumpRelayGeometry result;
        result.faceCount = mesh.numberOfCells * mesh.numberOfFacesPerCell;
        result.exitMask.assign(relays.size() * result.faceCount, 0u);
        result.descriptors.reserve(relays.size());
        for(std::size_t relayIndex = 0u; relayIndex < relays.size(); ++relayIndex)
        {
            auto const& relay = relays[relayIndex];
            auto const exitFrame = makeRelayFrame(mesh, relay.exitSurfaces);
            auto const entryFrame = makeRelayFrame(mesh, relay.entrySurfaces);
            PumpRelayDeviceDescriptor descriptor;
            descriptor.exitOrigin = exitFrame.origin;
            descriptor.exitU = exitFrame.u;
            descriptor.exitV = exitFrame.v;
            descriptor.exitNormal = exitFrame.normal;
            descriptor.entryOrigin = entryFrame.origin;
            descriptor.entryU = entryFrame.u;
            descriptor.entryV = entryFrame.v;
            descriptor.entryNormal = entryFrame.normal;
            descriptor.cosine = std::cos(relay.rotation);
            descriptor.sine = std::sin(relay.rotation);
            descriptor.offsetU = relay.offset[0u];
            descriptor.offsetV = relay.offset[1u];
            descriptor.tiltU = relay.tilt[0u];
            descriptor.tiltV = relay.tilt[1u];
            descriptor.magnification = relay.magnification;
            descriptor.transmission = relay.transmission;
            descriptor.flipU = relay.flipU ? -1 : 1;
            descriptor.flipV = relay.flipV ? -1 : 1;
            descriptor.entryFaceBegin = static_cast<unsigned>(result.entryFaceIds.size());
            for(auto const& face : entryFrame.faces)
                result.entryFaceIds.push_back(face.cell * mesh.numberOfFacesPerCell + face.localFace);
            descriptor.entryFaceEnd = static_cast<unsigned>(result.entryFaceIds.size());
            for(auto const& face : exitFrame.faces)
            {
                unsigned const faceId = face.cell * mesh.numberOfFacesPerCell + face.localFace;
                result.exitMask[relayIndex * result.faceCount + faceId] = 1u;
            }
            result.descriptors.push_back(descriptor);
        }
        return result;
    }

    /**
     * @param rays Host ray states to project into one component array.
     * @param getter Callable returning the requested component from one ray.
     * @return Component values in ray order; empty input produces one padding element.
     */
    template<typename T_Value>
    [[nodiscard]] inline std::vector<T_Value> pumpRayValues(
        std::vector<GeneralPumpRayState> const& rays,
        std::invocable<GeneralPumpRayState const&> auto getter)
    {
        std::vector<T_Value> result;
        result.reserve(rays.size());
        for(auto const& ray : rays)
            result.push_back(static_cast<T_Value>(getter(ray)));
        if(result.empty())
            result.emplace_back();
        return result;
    }

    /**
     * @param values Host values destined for device storage.
     * @return `values`, padded with one default element when empty.
     */
    template<typename T_Value>
    [[nodiscard]] inline std::vector<T_Value> pumpDeviceStorage(std::vector<T_Value> values)
    {
        if(values.empty())
            values.emplace_back();
        return values;
    }

    /** @brief Device-owned ray state, relay geometry, and cache for one pump source. */
    template<alpaka::onHost::concepts::Device T_Device>
    class GeneralPumpDeviceSource
    {
        using T_DoubleBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_SignedIndexBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::int32_t>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_DescriptorBuffer = ALPAKA_TYPEOF(
            alpaka::onHost::alloc<PumpRelayDeviceDescriptor>(std::declval<T_Device&>(), std::size_t{1u}));

        template<typename T_Value>
        [[nodiscard]] static auto makeExactCacheBuffer(concepts::Queue auto const& queue, std::size_t const extent)
        {
            using T_Buffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<T_Value>(queue.getDevice(), std::size_t{1u}));
            if constexpr(hase::core::compileTimeConfig::exactPumpCache)
                return std::optional<T_Buffer>{alpaka::onHost::alloc<T_Value>(queue.getDevice(), extent)};
            else
                return std::optional<T_Buffer>{};
        }

    public:
        /**
         * @brief Upload one sampled source and allocate its relay state.
         * @param queue Queue selecting the destination device.
         * @param rays Physical host ray states in partition order.
         * @param geometry Flattened relay geometry consumed by boundary policies.
         * @param pumpSteps Number of simulation steps during which the source is active.
         * @param rngSeed Source seed reused for deterministic reservoir passes.
         */
        GeneralPumpDeviceSource(
            concepts::Queue auto const& queue,
            std::vector<GeneralPumpRayState> const& rays,
            PumpRelayGeometry geometry,
            std::uint32_t const pumpSteps,
            std::uint32_t const rngSeed)
            : m_geometry(
                  hase::core::PositionBufferSoA<T_Device>{
                      queue,
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.position.x; }),
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.position.y; }),
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.position.z; })},
                  hase::core::DirectionBufferSoA<T_Device>{
                      queue,
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.direction.x; }),
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.direction.y; }),
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.direction.z; })})
            , m_power(
                  hase::alpakaUtils::toDevice(
                      queue,
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.power; })))
            , m_wavelength(
                  hase::alpakaUtils::toDevice(
                      queue,
                      pumpRayValues<double>(rays, [](GeneralPumpRayState const& ray) { return ray.wavelength; })))
            , m_cell(
                  hase::alpakaUtils::toDevice(
                      queue,
                      pumpRayValues<std::uint32_t>(rays, [](GeneralPumpRayState const& ray) { return ray.cell; })))
            , m_forbiddenFace(
                  hase::alpakaUtils::toDevice(
                      queue,
                      pumpRayValues<std::int32_t>(
                          rays,
                          [](GeneralPumpRayState const& ray) { return ray.forbiddenFace; })))
            , m_descriptors(hase::alpakaUtils::toDevice(queue, pumpDeviceStorage(geometry.descriptors)))
            , m_exitMask(hase::alpakaUtils::toDevice(queue, pumpDeviceStorage(geometry.exitMask)))
            , m_entryFaceIds(hase::alpakaUtils::toDevice(queue, pumpDeviceStorage(geometry.entryFaceIds)))
            , m_cacheState(
                  makeExactCacheBuffer<std::uint32_t>(
                      queue,
                      std::max<std::size_t>(1u, geometry.descriptors.size() * rays.size())))
            , m_cacheTargetFace(
                  makeExactCacheBuffer<std::uint32_t>(
                      queue,
                      std::max<std::size_t>(1u, geometry.descriptors.size() * rays.size())))
            , m_cacheBarycentric0(
                  makeExactCacheBuffer<double>(
                      queue,
                      std::max<std::size_t>(1u, geometry.descriptors.size() * rays.size())))
            , m_cacheBarycentric1(
                  makeExactCacheBuffer<double>(
                      queue,
                      std::max<std::size_t>(1u, geometry.descriptors.size() * rays.size())))
            , m_cacheBarycentric2(
                  makeExactCacheBuffer<double>(
                      queue,
                      std::max<std::size_t>(1u, geometry.descriptors.size() * rays.size())))
            , m_surfaceReservoirScratch(
                  hase::core::compileTimeConfig::exactPumpCache || geometry.descriptors.empty() || rays.empty()
                      ? nullptr
                      : std::make_unique<hase::core::SurfaceReservoirScratch<T_Device>>(
                            queue.getDevice(),
                            geometry.faceCount,
                            pumpReservoirSlots,
                            static_cast<std::uint32_t>(rays.size())))
            , m_faceCount{geometry.faceCount}
            , m_relayCount{static_cast<std::uint32_t>(geometry.descriptors.size())}
            , m_rayCount{static_cast<std::uint32_t>(rays.size())}
            , m_pumpSteps{pumpSteps}
            , m_rngSeed{rngSeed}
        {
            if constexpr(hase::core::compileTimeConfig::exactPumpCache)
            {
                auto const cacheExtent
                    = alpaka::Vec{std::max<std::size_t>(1u, geometry.descriptors.size() * rays.size())};
                alpaka::onHost::fill(queue, *m_cacheState, 0u, cacheExtent);
                alpaka::onHost::fill(queue, *m_cacheTargetFace, 0u, cacheExtent);
                alpaka::onHost::fill(queue, *m_cacheBarycentric0, 0.0, cacheExtent);
                alpaka::onHost::fill(queue, *m_cacheBarycentric1, 0.0, cacheExtent);
                alpaka::onHost::fill(queue, *m_cacheBarycentric2, 0.0, cacheExtent);
            }
        }

        /** @return Number of physical rays represented by the padded device storage. */
        [[nodiscard]] unsigned rayCount() const
        {
            return m_rayCount;
        }

        /**
         * @param simulationStep Zero-based physical simulation step.
         * @return Whether this source pumps during that step.
         */
        [[nodiscard]] bool active(unsigned const simulationStep) const
        {
            return simulationStep < m_pumpSteps;
        }

        /** @return Device buffer recording per-relay cached face mappings. */
        [[nodiscard]] auto const& cacheState() const
        {
            return *m_cacheState;
        }

        /**
         * @brief Enqueue this source with finite planar relay mappings.
         * @param devBundle Device and executor used for the ray launch.
         * @param queue Queue receiving fill and trace operations.
         * @param mesh Device-resident trace view.
         * @param betaVolume Current cell excitation fractions.
         * @param vertexPumpIntegral Material-vertex photon-exchange accumulator.
         */
        template<alpaka::concepts::Executor T_Executor>
        requires(hase::core::compileTimeConfig::exactPumpCache)
        void enqueue(
            hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            hase::data::TraceView const mesh,
            alpaka::concepts::IBuffer<double> auto& betaVolume,
            alpaka::concepts::IBuffer<double> auto& vertexPumpIntegral,
            pumpBoundaryPolicy::Relay)
        {
            enqueueWithFactory(
                devBundle,
                queue,
                mesh,
                betaVolume,
                vertexPumpIntegral,
                DevicePumpRelayBoundaryFactory{
                    m_descriptors.getView(),
                    m_exitMask.getView(),
                    m_entryFaceIds.getView(),
                    m_cacheState->getView(),
                    m_cacheTargetFace->getView(),
                    m_cacheBarycentric0->getView(),
                    m_cacheBarycentric1->getView(),
                    m_cacheBarycentric2->getView(),
                    m_faceCount,
                    m_relayCount,
                    m_rayCount});
        }

        /**
         * @brief Enqueue this source with barycentric surface-reservoir recirculation.
         * @param devBundle Device and executor used for traces and scans.
         * @param queue Queue receiving all reservoir passes.
         * @param mesh Device-resident trace view.
         * @param betaVolume Current cell excitation fractions.
         * @param vertexPumpIntegral Material-vertex photon-exchange accumulator.
         */
        template<alpaka::concepts::Executor T_Executor>
        requires(!core::compileTimeConfig::exactPumpCache)
        void enqueue(
            hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            hase::data::TraceView const mesh,
            alpaka::concepts::IBuffer<double> auto& betaVolume,
            alpaka::concepts::IBuffer<double> auto& vertexPumpIntegral,
            pumpBoundaryPolicy::SrmBarycentric)
        {
            if(m_rayCount == 0u)
                return;
            if(m_surfaceReservoirScratch == nullptr)
            {
                enqueueWithFactory(
                    devBundle,
                    queue,
                    mesh,
                    betaVolume,
                    vertexPumpIntegral,
                    StorePumpSrmBoundaryFactory{});
                return;
            }

            auto& scratch = *m_surfaceReservoirScratch;
            scratch.validate(m_faceCount, m_rayCount);
            scratch.clear(queue, scratch.reservoir.first);
            scratch.clear(queue, scratch.reservoir.second);
            alpaka::onHost::wait(queue);
            enqueueWithFactory(
                devBundle,
                queue,
                mesh,
                betaVolume,
                vertexPumpIntegral,
                PumpSurfaceReservoirBoundaryFactory{
                    m_descriptors.getView(),
                    m_exitMask.getView(),
                    scratch.reservoir.first.view(pumpReservoirSlots),
                    m_faceCount,
                    0u,
                    m_rngSeed,
                    0u,
                    true});
            alpaka::onHost::wait(queue);

            bool inputFirst = true;
            double previousWeight = scratch.updateSampling(
                devBundle,
                queue,
                scratch.reservoir.first,
                pumpReservoirSlots,
                m_rayCount,
                m_rngSeed,
                0u);
            auto const frameSpec = hase::alpakaUtils::getFrameSpec<std::uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{m_rayCount});
            for(std::uint32_t pass = 1u; pass <= m_relayCount && previousWeight > 0.0; ++pass)
            {
                bool const captureOutput = pass < m_relayCount;
                auto& input = inputFirst ? scratch.reservoir.first : scratch.reservoir.second;
                auto& output = inputFirst ? scratch.reservoir.second : scratch.reservoir.first;
                if(captureOutput)
                {
                    scratch.clear(queue, output);
                    alpaka::onHost::wait(queue);
                }
                queue.enqueue(
                    frameSpec,
                    alpaka::KernelBundle{
                        TracePumpSurfaceReservoir{},
                        mesh,
                        betaVolume,
                        input.view(pumpReservoirSlots),
                        scratch.samplingView(m_faceCount <= m_rayCount),
                        output.view(pumpReservoirSlots),
                        m_descriptors.getView(),
                        m_exitMask.getView(),
                        vertexPumpIntegral,
                        m_faceCount,
                        m_rayCount,
                        pass,
                        previousWeight / static_cast<double>(m_rayCount),
                        m_rngSeed,
                        pass,
                        captureOutput});
                alpaka::onHost::wait(queue);
                if(captureOutput)
                {
                    inputFirst = !inputFirst;
                    previousWeight = scratch.updateSampling(
                        devBundle,
                        queue,
                        output,
                        pumpReservoirSlots,
                        m_rayCount,
                        m_rngSeed,
                        pass);
                }
            }
        }

    private:
        template<alpaka::concepts::Executor T_Executor>
        void enqueueWithFactory(
            hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            hase::data::TraceView const mesh,
            alpaka::concepts::IBuffer<double> auto& betaVolume,
            alpaka::concepts::IBuffer<double> auto& vertexPumpIntegral,
            std::invocable<unsigned> auto boundaryPolicyFactory)
        {
            if(m_rayCount == 0u)
                return;
            auto frameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{m_rayCount});
            queue.enqueue(
                frameSpec,
                alpaka::KernelBundle{
                    TraceGeneralPump{},
                    mesh,
                    betaVolume,
                    m_geometry.view(),
                    m_power,
                    m_wavelength,
                    m_cell,
                    m_forbiddenFace,
                    boundaryPolicyFactory,
                    vertexPumpIntegral,
                    m_rayCount});
        }

        hase::core::RayGeometryBufferSoA<T_Device> m_geometry;
        T_DoubleBuffer m_power, m_wavelength;
        T_UnsignedBuffer m_cell;
        T_SignedIndexBuffer m_forbiddenFace;
        T_DescriptorBuffer m_descriptors;
        T_UnsignedBuffer m_exitMask, m_entryFaceIds;
        std::optional<T_UnsignedBuffer> m_cacheState, m_cacheTargetFace;
        std::optional<T_DoubleBuffer> m_cacheBarycentric0, m_cacheBarycentric1, m_cacheBarycentric2;
        std::unique_ptr<hase::core::SurfaceReservoirScratch<T_Device>> m_surfaceReservoirScratch;
        std::uint32_t m_faceCount = 0u;
        std::uint32_t m_relayCount = 0u;
        std::uint32_t m_rayCount = 0u;
        std::uint32_t m_pumpSteps = 0u;
        std::uint32_t m_rngSeed = 0u;
    };

    /**
     * @brief Sample and upload every registered pump source for one worker partition.
     * @param queue Queue selecting the destination device.
     * @param mesh Prepared host trace.
     * @param pump Collection of prepared pump sources.
     * @param firstRay First global ray assigned to this worker for every source.
     * @param localRayCount Maximum source rays assigned to this worker.
     * @return Device-owned sources in registration order.
     */
    template<alpaka::onHost::concepts::Device T_Device>
    [[nodiscard]] inline std::vector<GeneralPumpDeviceSource<T_Device>> prepareGeneralPumpDeviceSources(
        concepts::Queue auto const& queue,
        hase::data::TraceData const& mesh,
        hase::core::PumpParameters const& pump,
        unsigned const firstRay = 0u,
        unsigned const localRayCount = std::numeric_limits<unsigned>::max())
    {
        std::vector<GeneralPumpDeviceSource<T_Device>> result;
        result.reserve(pump.sources.size());
        for(auto const& source : pump.sources)
        {
            if(source.pumpSteps == 0u)
            {
                result
                    .emplace_back(queue, std::vector<GeneralPumpRayState>{}, PumpRelayGeometry{}, 0u, source.rngSeed);
                continue;
            }
            auto rays = samplePumpSource(mesh, source, source.rayCount, source.rngSeed, firstRay, localRayCount);
            result.emplace_back(
                queue,
                rays,
                preparePumpRelayGeometry(mesh, source.relays),
                source.pumpSteps,
                source.rngSeed);
        }
        return result;
    }

    /** @brief Kernel converting material-vertex pump integrals into cell rates. */
    struct ProjectVertexPumpRateToCells
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            hase::data::TraceView const mesh,
            alpaka::concepts::IView<double> auto vertexIntegral,
            alpaka::concepts::IView<double> auto cellRate) const
        {
            for(auto [cell] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{mesh.numberOfCells}))
            {
                if(!mesh.isActive(cell))
                {
                    cellRate[cell] = 0.0;
                    continue;
                }

                double rateSum = 0.0;
                for(unsigned localVertex = 0u; localVertex < mesh.numberOfCellVertices; ++localVertex)
                {
                    unsigned const materialVertex
                        = mesh.getMaterialId(cell) * mesh.numberOfMeshPoints
                          + mesh.cellPointIndices[cell * mesh.numberOfCellVertices + localVertex];
                    double const volume = mesh.lumpedMaterialVertexVolumes[materialVertex];
                    rateSum += volume > 0.0 ? vertexIntegral[materialVertex] / volume : 0.0;
                }
                // Volume averaging the four lumped vertex rates preserves the deposited
                // integral: sum_cell(V_cell * cellRate) == sum_vertex(vertexIntegral).
                cellRate[cell] = rateSum / static_cast<double>(mesh.numberOfCellVertices);
            }
        }
    };

    /**
     * @brief Clear and accumulate pump photon exchange at material vertices.
     * @param devBundle Device and executor used for source launches.
     * @param queue Queue receiving all source operations.
     * @param mesh Device-resident trace view.
     * @param sources Prepared device sources in registration order.
     * @param betaVolume Current cell excitation fractions.
     * @param vertexPumpIntegral Output material-vertex integrals, cleared first.
     * @param simulationStep Zero-based physical simulation step.
     * @param boundaryPolicy Relay or surface-reservoir boundary selector.
     */
    template<
        alpaka::onHost::concepts::Device T_Device,
        alpaka::concepts::Executor T_Executor,
        typename T_BoundaryPolicy = std::conditional_t<
            hase::core::compileTimeConfig::exactPumpCache,
            pumpBoundaryPolicy::Relay,
            pumpBoundaryPolicy::SrmBarycentric>>
    void enqueueGeneralPumpIntegrals(
        hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
        concepts::Queue auto const& queue,
        hase::data::TraceView const mesh,
        std::vector<GeneralPumpDeviceSource<T_Device>>& sources,
        alpaka::concepts::IBuffer<double> auto& betaVolume,
        alpaka::concepts::IBuffer<double> auto& vertexPumpIntegral,
        unsigned const simulationStep,
        T_BoundaryPolicy boundaryPolicy = {})
    {
        HASE_HOST_ROUTINE_SCOPE("pump.enqueue_integrals");
        alpaka::onHost::fill(
            queue,
            vertexPumpIntegral,
            0.0,
            alpaka::Vec{static_cast<std::size_t>(mesh.numberOfMaterials) * mesh.numberOfMeshPoints});
        for(auto& source : sources)
        {
            if(source.active(simulationStep))
                source.enqueue(devBundle, queue, mesh, betaVolume, vertexPumpIntegral, boundaryPolicy);
        }
    }

    /**
     * @brief Enqueue conservative projection from vertex integrals to cell rates.
     * @param devBundle Device and executor used for the cell launch.
     * @param queue Queue receiving the projection kernel.
     * @param mesh Device trace containing lumped material-vertex volumes.
     * @param vertexPumpIntegral Material-vertex photon-exchange integrals.
     * @param cellRate Output pump population rate per cell.
     */
    template<alpaka::onHost::concepts::Device T_Device, alpaka::concepts::Executor T_Executor>
    void enqueueProjectVertexPumpRateToCells(
        hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
        concepts::Queue auto const& queue,
        hase::data::TraceView const mesh,
        alpaka::concepts::IBuffer<double> auto& vertexPumpIntegral,
        alpaka::concepts::IBuffer<double> auto& cellRate)
    {
        HASE_HOST_ROUTINE_SCOPE("pump.enqueue_project_vertices");
        auto cellFrameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
            devBundle.device,
            devBundle.executor,
            alpaka::Vec{mesh.numberOfCells});
        queue.enqueue(
            cellFrameSpec,
            alpaka::KernelBundle{ProjectVertexPumpRateToCells{}, mesh, vertexPumpIntegral, cellRate});
    }

    /**
     * @brief Enqueue complete pump transport and conservative cell-rate projection.
     * @param devBundle Device and executor used for all launches.
     * @param queue Queue receiving transport and projection work.
     * @param mesh Device-resident trace view.
     * @param sources Prepared device sources.
     * @param betaVolume Current cell excitation fractions.
     * @param vertexPumpIntegral Temporary material-vertex integral storage.
     * @param cellRate Output pump population rate per cell.
     * @param simulationStep Zero-based physical simulation step.
     * @param boundaryPolicy Relay or surface-reservoir boundary selector.
     */
    template<
        alpaka::onHost::concepts::Device T_Device,
        alpaka::concepts::Executor T_Executor,
        typename T_BoundaryPolicy = std::conditional_t<
            hase::core::compileTimeConfig::exactPumpCache,
            pumpBoundaryPolicy::Relay,
            pumpBoundaryPolicy::SrmBarycentric>>
    void enqueueGeneralPump(
        hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
        concepts::Queue auto const& queue,
        hase::data::TraceView const mesh,
        std::vector<GeneralPumpDeviceSource<T_Device>>& sources,
        alpaka::concepts::IBuffer<double> auto& betaVolume,
        alpaka::concepts::IBuffer<double> auto& vertexPumpIntegral,
        alpaka::concepts::IBuffer<double> auto& cellRate,
        unsigned const simulationStep,
        T_BoundaryPolicy boundaryPolicy = {})
    {
        enqueueGeneralPumpIntegrals(
            devBundle,
            queue,
            mesh,
            sources,
            betaVolume,
            vertexPumpIntegral,
            simulationStep,
            boundaryPolicy);
        enqueueProjectVertexPumpRateToCells(devBundle, queue, mesh, vertexPumpIntegral, cellRate);
    }
} // namespace hase::kernels
