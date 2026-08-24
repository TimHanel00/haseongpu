/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <data/TraceData.hpp>
#include <kernels/forward/barycentric.hpp>

#include <limits>

namespace hase::kernels::forward
{
    inline constexpr double barycentricTraversalTolerance = 64.0 * std::numeric_limits<double>::epsilon();

    /** @brief Nearest positive Tet4 face intersection and tied-face mask. */
    struct Tet4FaceIntersection
    {
        int localFace = -1;
        double length = std::numeric_limits<double>::max();
        unsigned tiedFaceMask = 0u;
    };

    /**
     * @param point Segment origin.
     * @param direction Segment direction.
     * @param length Signed travel distance.
     * @return `point + direction * length`.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC hase::core::Point advance(
        hase::core::Point const point,
        hase::core::Point const direction,
        double const length)
    {
        return point + direction * length;
    }

    /** @return Unit vector parallel to `value`, or zero for negligible length. */
    [[nodiscard]] inline ALPAKA_FN_ACC hase::core::Point normalize(hase::core::Point const value)
    {
        double const length = value.euclidLength();
        if(length <= std::numeric_limits<double>::epsilon())
        {
            return hase::core::Point{0.0, 0.0, 0.0};
        }
        return value * (1.0 / length);
    }

    /**
     * @param mesh Device trace containing Tet4 face connectivity.
     * @param tet Cell index.
     * @param localFace Local face index.
     * @return Arithmetic centroid of the three face vertices.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC hase::core::Point faceCentroid(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        unsigned const localFace)
    {
        hase::core::Point sum{0.0, 0.0, 0.0};
        for(unsigned localVertex = 0u; localVertex < hase::data::tet4FaceWidth; ++localVertex)
        {
            int const point = mesh.getCellFacePoint(tet, localFace, localVertex);
            if(point < 0)
            {
                return sum;
            }
            sum = sum + mesh.getPoint(static_cast<unsigned>(point));
        }
        return sum * (1.0 / static_cast<double>(hase::data::tet4FaceWidth));
    }

    /**
     * @param mesh Device trace containing Tet4 geometry.
     * @param tet Cell index.
     * @param localFace Local face index.
     * @return Unit face normal oriented away from the cell center.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC hase::core::Point outwardFaceNormal(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        unsigned const localFace)
    {
        int const p0 = mesh.getCellFacePoint(tet, localFace, 0u);
        int const p1 = mesh.getCellFacePoint(tet, localFace, 1u);
        int const p2 = mesh.getCellFacePoint(tet, localFace, 2u);
        if(p0 < 0 || p1 < 0 || p2 < 0)
        {
            return hase::core::Point{0.0, 0.0, 0.0};
        }
        hase::core::Point const a = mesh.getPoint(static_cast<unsigned>(p0));
        hase::core::Point const b = mesh.getPoint(static_cast<unsigned>(p1));
        hase::core::Point const c = mesh.getPoint(static_cast<unsigned>(p2));
        hase::core::Point normal = normalize(hase::core::cross(b - a, c - a));
        hase::core::Point const centroid = (a + b + c) * (1.0 / 3.0);
        if(hase::core::dot(normal, mesh.getCellCenterPoint(tet) - centroid) > 0.0)
        {
            normal = normal * -1.0;
        }
        return normal;
    }

    /**
     * @param direction Incident direction.
     * @param outwardNormal Unit outward surface normal.
     * @return Normalized specular-reflection direction.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC hase::core::Point reflectedDirection(
        hase::core::Point const direction,
        hase::core::Point const outwardNormal)
    {
        return normalize(direction - outwardNormal * (2.0 * hase::core::dot(direction, outwardNormal)));
    }

    /**
     * @param coordinate Current affine face coordinate.
     * @param directionalChange Coordinate change per unit ray length.
     * @param maxLength Largest accepted positive length.
     * @return Positive intersection length, or zero when the face is not crossed.
     */
    [[nodiscard]] inline ALPAKA_FN_HOST_ACC double barycentricFaceIntersectionLength(
        double const coordinate,
        double const directionalChange,
        double const maxLength)
    {
        if(directionalChange >= 0.0)
        {
            return 0.0;
        }
        double const length = -coordinate / directionalChange;
        return length > 0.0 && length <= maxLength ? length : 0.0;
    }

    /**
     * @param mesh Device trace containing affine face planes.
     * @param tet Current cell.
     * @param origin Current ray position.
     * @param direction Ray direction.
     * @param forbiddenFace Entry face excluded from selection.
     * @return Nearest positive face intersection, including numerically tied faces.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC Tet4FaceIntersection nextFaceIntersection(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        hase::core::Point const origin,
        hase::core::Point const direction,
        int const forbiddenFace)
    {
        alpaka::Vec<double, hase::data::tet4FaceCount> candidates{0.0, 0.0, 0.0, 0.0};
        Tet4FaceIntersection result;

        // The first decreasing face coordinate to reach zero is the Tet4 exit face.
        for(unsigned localFace = 0u; localFace < mesh.numberOfFacesPerCell; ++localFace)
        {
            if(static_cast<int>(localFace) == forbiddenFace)
            {
                continue;
            }
            double const coordinate = mesh.getFaceBarycentricCoordinate(tet, localFace, origin);
            double const directionalChange = mesh.getFaceBarycentricDirection(tet, localFace, direction);
            candidates[localFace]
                = barycentricFaceIntersectionLength(coordinate, directionalChange, std::numeric_limits<double>::max());
            if(candidates[localFace] > 0.0 && candidates[localFace] < result.length)
            {
                result.length = candidates[localFace];
                result.localFace = static_cast<int>(localFace);
            }
        }

        if(result.localFace < 0)
        {
            return result;
        }

        for(unsigned localFace = 0u; localFace < mesh.numberOfFacesPerCell; ++localFace)
        {
            if(static_cast<int>(localFace) == forbiddenFace || candidates[localFace] <= 0.0)
            {
                continue;
            }
            double const coordinate = mesh.getFaceBarycentricCoordinate(tet, localFace, origin);
            double const directionalChange = mesh.getFaceBarycentricDirection(tet, localFace, direction);
            double const coordinateAtIntersection = coordinate + result.length * directionalChange;
            if(alpaka::math::abs(coordinateAtIntersection) <= barycentricTraversalTolerance)
            {
                result.tiedFaceMask |= 1u << localFace;
            }
        }
        result.tiedFaceMask |= 1u << static_cast<unsigned>(result.localFace);
        for(unsigned localFace = 0u; localFace < mesh.numberOfFacesPerCell; ++localFace)
        {
            if((result.tiedFaceMask & (1u << localFace)) != 0u)
            {
                result.localFace = static_cast<int>(localFace);
                break;
            }
        }
        return result;
    }

    /** @return Whether `tiedFaceMask` contains more than one face bit. */
    [[nodiscard]] inline ALPAKA_FN_ACC bool hasMultipleTiedFaces(unsigned const tiedFaceMask)
    {
        return tiedFaceMask != 0u && (tiedFaceMask & (tiedFaceMask - 1u)) != 0u;
    }

    /**
     * @brief Resolve attenuation or gain from the material of the current cell.
     *
     * Cross sections are interpolated at the ray wavelength on the device.
     * This keeps propagation valid when a ray enters a cell owned by another
     * material without copying or rebinding a launch-global spectrum.
     *
     * @param mesh Device trace containing cell material and excitation fields.
     * @param tet Current cell index.
     * @param wavelength Ray wavelength in metres.
     * @return Net local gain coefficient, including bulk attenuation, in inverse metres.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC double localGainCoefficient(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        double const wavelength)
    {
        auto const crossSections = mesh.crossSectionsForCell(tet, wavelength);
        double const stimulatedCoefficient
            = mesh.isActive(tet) ? mesh.activeIonDensity(tet)
                                       * (mesh.getBetaVolume(tet) * (crossSections.emission + crossSections.absorption)
                                          - crossSections.absorption)
                                 : 0.0;
        return stimulatedCoefficient - mesh.bulkAttenuation(tet);
    }

    /**
     * @param mesh Device trace containing local optical coefficients.
     * @param tet Current cell index.
     * @param length Traversed segment length in metres.
     * @param wavelength Ray wavelength in metres.
     * @return Multiplicative power gain `exp(g * length)`.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC double localSegmentGain(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        double const length,
        double const wavelength)
    {
        return alpaka::math::exp(localGainCoefficient(mesh, tet, wavelength) * length);
    }

    /**
     * @param mesh Device trace containing local optical coefficients.
     * @param tet Current cell index.
     * @param length Traversed segment length in metres.
     * @param wavelength Ray wavelength in metres.
     * @return Integral of exponential gain along the segment, in metres.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC double localSegmentTrackLengthIntegral(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        double const length,
        double const wavelength)
    {
        double const gainCoefficient = localGainCoefficient(mesh, tet, wavelength);
        double const gainLength = gainCoefficient * length;
        if(alpaka::math::abs(gainLength) < 1.0e-8)
        {
            return length;
        }
        return (alpaka::math::exp(gainLength) - 1.0) / gainCoefficient;
    }

    /**
     * @param mesh Device trace containing Tet4 geometry.
     * @param tet Current cell index.
     * @param midpoint Segment midpoint.
     * @return Center-proximity weight of the midpoint in `[0, 1]`.
     */
    [[nodiscard]] inline ALPAKA_FN_ACC double segmentCenterWeight(
        hase::data::TraceView const& mesh,
        unsigned const tet,
        hase::core::Point const midpoint)
    {
        return centerProximityWeight(barycentricCoordinates(mesh, tet, midpoint));
    }
} // namespace hase::kernels::forward
