/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <data/TraceData.hpp>
#include <kernels/forward/rayWalk.hpp>

namespace hase::kernels::forward
{
    struct BoundaryInteraction
    {
        hase::core::Direction reflected;
        hase::core::Direction transmitted;
        double reflectance{};
        bool totalInternalReflection{};
    };

    /** @brief Deterministically weighted children produced by one boundary hit. */
    struct BoundaryBranchWeights
    {
        double reflected{};
        double transmitted{};
    };

    /** @brief Resolve specular reflection and Snell transmission for one interface hit. */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC BoundaryInteraction boundaryInteraction(
        hase::core::Direction const direction,
        hase::core::Direction const outwardNormal,
        double const sourceRefractiveIndex,
        double const targetRefractiveIndex,
        double const configuredReflectivity)
    {
        auto const incident = normalize(direction);
        auto const normal = normalize(outwardNormal);
        double const cosine = alpaka::math::max(0.0, alpaka::math::min(1.0, hase::core::dot(incident, normal)));
        double const ratio = sourceRefractiveIndex / targetRefractiveIndex;
        double const discriminant = 1.0 - ratio * ratio * alpaka::math::max(0.0, 1.0 - cosine * cosine);
        bool const totalInternalReflection
            = sourceRefractiveIndex <= 0.0 || targetRefractiveIndex <= 0.0 || discriminant < 0.0;
        return BoundaryInteraction{
            reflectedDirection(incident, normal),
            totalInternalReflection
                ? hase::core::Direction{0.0, 0.0, 0.0}
                : normalize(incident * ratio + normal * (alpaka::math::sqrt(discriminant) - ratio * cosine)),
            totalInternalReflection ? 1.0 : alpaka::math::max(0.0, alpaka::math::min(1.0, configuredReflectivity)),
            totalInternalReflection};
    }

    /** @brief Split one boundary weight without stochastic branch selection. */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC BoundaryBranchWeights splitBoundaryWeights(
        double const boundaryWeight,
        BoundaryInteraction const& interaction,
        bool const hasTransmission,
        bool const useReflections)
    {
        return {
            useReflections ? boundaryWeight * interaction.reflectance : 0.0,
            hasTransmission ? boundaryWeight * (1.0 - interaction.reflectance) : 0.0};
    }

    /**
     * @brief Return the SRM reflection weight for an outward boundary hit.
     *
     * The forward surface-resampling method relaunches reflected rays from
     * this boundary and continues along the reflected forward direction.
     * @param mesh Trace view containing boundary optics and material indices.
     * @param tet Incident cell index.
     * @param localFace Cell-local boundary face index.
     * @param direction Incident ray direction.
     * @param outwardNormal Unit face normal pointing out of `tet`.
     * @return One for total internal reflection, otherwise the non-negative
     * configured constant reflectivity; zero for a non-boundary face.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC double boundaryReflectance(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        unsigned const localFace,
        hase::core::Point const direction,
        hase::core::Point const outwardNormal)
    {
        int const boundary = mesh.cellFaceBoundaries[tet * mesh.numberOfFacesPerCell + localFace];
        if(boundary <= 0)
        {
            return 0.0;
        }
        return boundaryInteraction(
                   direction,
                   outwardNormal,
                   static_cast<double>(mesh.getSurfaceRefractiveIndexInside(tet, localFace)),
                   static_cast<double>(mesh.getSurfaceRefractiveIndexOutside(tet, localFace)),
                   static_cast<double>(mesh.getSurfaceReflectivity(tet, localFace)))
            .reflectance;
    }
} // namespace hase::kernels::forward
