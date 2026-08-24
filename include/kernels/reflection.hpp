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
        double const nInside = static_cast<double>(mesh.getSurfaceRefractiveIndexInside(tet, localFace));
        double const nOutside = static_cast<double>(mesh.getSurfaceRefractiveIndexOutside(tet, localFace));
        if(nInside > 0.0 && nOutside > 0.0)
        {
            double const cosIncident
                = alpaka::math::min(1.0, alpaka::math::abs(hase::core::dot(normalize(direction), outwardNormal)));
            double const sin2Incident = alpaka::math::max(0.0, 1.0 - cosIncident * cosIncident);
            double const ratio = nInside / nOutside;
            if(ratio * ratio * sin2Incident > 1.0)
            {
                return 1.0;
            }
        }
        return alpaka::math::max(0.0, static_cast<double>(mesh.getSurfaceReflectivity(tet, localFace)));
    }
} // namespace hase::kernels::forward
