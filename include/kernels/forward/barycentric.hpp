/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <core/geometry.hpp>

#include <array>
#include <cmath>
#include <limits>

namespace hase::kernels::forward
{
    using BarycentricTet4 = std::array<double, 4u>;

    /**
     * @param a First Tet4 vertex.
     * @param b Second Tet4 vertex.
     * @param c Third Tet4 vertex.
     * @param d Fourth Tet4 vertex.
     * @return Six times the oriented tetrahedron volume.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC double signedTetVolume6(
        hase::core::Point const a,
        hase::core::Point const b,
        hase::core::Point const c,
        hase::core::Point const d)
    {
        return hase::core::dot(hase::core::cross(b - a, c - a), d - a);
    }

    /**
     * @param point Cartesian point to express in Tet4 coordinates.
     * @param a First Tet4 vertex.
     * @param b Second Tet4 vertex.
     * @param c Third Tet4 vertex.
     * @param d Fourth Tet4 vertex.
     * @return Four affine barycentric coordinates, or equal weights for a degenerate Tet4.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC BarycentricTet4 barycentricCoordinates(
        hase::core::Point const point,
        hase::core::Point const a,
        hase::core::Point const b,
        hase::core::Point const c,
        hase::core::Point const d)
    {
        double const denominator = signedTetVolume6(a, b, c, d);
        if(alpaka::math::abs(denominator) <= std::numeric_limits<double>::epsilon())
        {
            return {0.25, 0.25, 0.25, 0.25};
        }

        double const invDenominator = 1.0 / denominator;
        return {
            signedTetVolume6(point, b, c, d) * invDenominator,
            signedTetVolume6(a, point, c, d) * invDenominator,
            signedTetVolume6(a, b, point, d) * invDenominator,
            signedTetVolume6(a, b, c, point) * invDenominator};
    }

    /**
     * @param mesh Device trace containing Tet4 connectivity.
     * @param tet Cell index of the Tet4.
     * @param point Cartesian point to express.
     * @return Four affine barycentric coordinates in local-vertex order.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC BarycentricTet4 barycentricCoordinates(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        hase::core::Point const point)
    {
        return barycentricCoordinates(
            point,
            mesh.getCellPoint(tet, 0u),
            mesh.getCellPoint(tet, 1u),
            mesh.getCellPoint(tet, 2u),
            mesh.getCellPoint(tet, 3u));
    }

    /**
     * @param mesh Device trace containing Tet4 connectivity.
     * @param tet Cell index of the Tet4.
     * @param point Cartesian contribution point.
     * @return Non-negative vertex weights normalized to sum to one.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC BarycentricTet4 normalizedBarycentricVertexWeights(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        hase::core::Point const point)
    {
        auto weights = barycentricCoordinates(mesh, tet, point);
        double weightSum = 0.0;
        for(double& weight : weights)
        {
            weight = alpaka::math::max(0.0, weight);
            weightSum += weight;
        }

        if(weightSum <= std::numeric_limits<double>::epsilon())
            return {0.25, 0.25, 0.25, 0.25};

        double const inverseWeightSum = 1.0 / weightSum;
        for(double& weight : weights)
            weight *= inverseWeightSum;
        return weights;
    }

    /**
     * @param mesh Device trace containing Tet4 connectivity.
     * @param tet Cell index of the traversed Tet4.
     * @param position Segment origin.
     * @param direction Unit segment direction.
     * @param length Segment length.
     * @return Normalized barycentric weights at the segment midpoint.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC BarycentricTet4 segmentMidpointBarycentricVertexWeights(
        data::TraceView const& mesh,
        unsigned const tet,
        core::Point const position,
        core::Point const direction,
        double const length)
    {
        auto const midpoint = position + direction * (0.5 * length);
        return normalizedBarycentricVertexWeights(mesh, tet, midpoint);
    }

    /**
     * @param barycentric Four Tet4 barycentric coordinates.
     * @return Linearized proximity to the Tet4 center, clamped to `[0, 1]`.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC double centerProximityWeight(BarycentricTet4 const& barycentric)
    {
        double distanceSquared = 0.0;
        for(double const coordinate : barycentric)
        {
            double const delta = coordinate - 0.25;
            distanceSquared += delta * delta;
        }
        constexpr double maxCenterDistance = 0.86602540378443864676;
        double const normalizedDistance = alpaka::math::sqrt(distanceSquared) / maxCenterDistance;
        return alpaka::math::max(0.0, 1.0 - normalizedDistance);
    }
} // namespace hase::kernels::forward
