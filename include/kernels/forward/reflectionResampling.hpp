/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <core/geometry.hpp>
#include <data/TraceData.hpp>
#include <kernels/forward/policyRay.hpp>

#include <cstdint>
#include <limits>

namespace hase::kernels::forward
{
    inline constexpr std::uint32_t invalidPreparedRayCell = std::numeric_limits<std::uint32_t>::max();
    inline constexpr std::uint32_t preparedRayTraceComponentCount = 7u;
    inline constexpr std::uint32_t preparedRayTraceTopologyCount = 2u;
    inline constexpr std::uint32_t reflectionCandidateComponentCount = 7u;

    /** @brief Prepared launch state consumed by one reflected forward-ASE history. */
    struct PreparedRayTraceState
    {
        hase::core::Point position;
        hase::core::Direction direction;
        double wavelength;
        std::uint32_t cell;
        std::int32_t forbiddenFace;
    };

    /** @brief Geometry and spectrum state retained for one reflected boundary candidate. */
    struct ReflectionCandidateState
    {
        hase::core::Point position;
        hase::core::Direction direction;
        double wavelength;
        std::uint32_t faceId;
    };

    /** @brief Ray-indexed reflected boundary candidates for one transport pass. */
    template<
        alpaka::concepts::IView<double> T_Components,
        alpaka::concepts::IView<double> T_Weights,
        alpaka::concepts::IView<std::uint32_t> T_FaceIds>
    struct ReflectionCandidateSpans
    {
        ALPAKA_FN_HOST_ACC void store(
            std::uint32_t const candidateIndex,
            ReflectionCandidateState const& candidate,
            double const weight)
        {
            components[0u * stride + candidateIndex] = candidate.position.x;
            components[1u * stride + candidateIndex] = candidate.position.y;
            components[2u * stride + candidateIndex] = candidate.position.z;
            components[3u * stride + candidateIndex] = candidate.direction.x;
            components[4u * stride + candidateIndex] = candidate.direction.y;
            components[5u * stride + candidateIndex] = candidate.direction.z;
            components[6u * stride + candidateIndex] = candidate.wavelength;
            weights[candidateIndex] = weight;
            faceIds[candidateIndex] = candidate.faceId;
        }

        [[nodiscard]] ALPAKA_FN_HOST_ACC ReflectionCandidateState load(std::uint32_t const candidateIndex) const
        {
            return {
                {components[0u * stride + candidateIndex],
                 components[1u * stride + candidateIndex],
                 components[2u * stride + candidateIndex]},
                {components[3u * stride + candidateIndex],
                 components[4u * stride + candidateIndex],
                 components[5u * stride + candidateIndex]},
                components[6u * stride + candidateIndex],
                faceIds[candidateIndex]};
        }

        T_Components components;
        T_Weights weights;
        T_FaceIds faceIds;
        std::uint32_t stride;
    };

    /** @brief Ordered cumulative weights and their total for reflected candidates. */
    template<alpaka::concepts::IView<double> T_Cdf, alpaka::concepts::IView<double> T_TotalWeight>
    struct ReflectionSamplingSpans
    {
        T_Cdf cdf;
        T_TotalWeight totalWeight;
    };

    /** @brief Device-resident launch state shared by the reflected forward-ASE subkernels. */
    template<alpaka::concepts::IView<double> T_Components, alpaka::concepts::IView<std::uint32_t> T_Topology>
    struct PreparedRayTraceSpans
    {
        ALPAKA_FN_HOST_ACC void store(std::uint32_t const rayIndex, PreparedRayTraceState const& ray)
        {
            components[0u * stride + rayIndex] = ray.position.x;
            components[1u * stride + rayIndex] = ray.position.y;
            components[2u * stride + rayIndex] = ray.position.z;
            components[3u * stride + rayIndex] = ray.direction.x;
            components[4u * stride + rayIndex] = ray.direction.y;
            components[5u * stride + rayIndex] = ray.direction.z;
            components[6u * stride + rayIndex] = ray.wavelength;
            topology[0u * stride + rayIndex] = ray.cell;
            topology[1u * stride + rayIndex] = static_cast<std::uint32_t>(ray.forbiddenFace + 1);
        }

        ALPAKA_FN_HOST_ACC void invalidate(std::uint32_t const rayIndex)
        {
            topology[0u * stride + rayIndex] = invalidPreparedRayCell;
        }

        [[nodiscard]] ALPAKA_FN_HOST_ACC PreparedRayTraceState load(std::uint32_t const rayIndex) const
        {
            return {
                {components[0u * stride + rayIndex],
                 components[1u * stride + rayIndex],
                 components[2u * stride + rayIndex]},
                {components[3u * stride + rayIndex],
                 components[4u * stride + rayIndex],
                 components[5u * stride + rayIndex]},
                components[6u * stride + rayIndex],
                topology[0u * stride + rayIndex],
                static_cast<std::int32_t>(topology[1u * stride + rayIndex]) - 1};
        }

        T_Components components;
        T_Topology topology;
        std::uint32_t stride;
    };

    /** @brief Reflected boundary state offered to the next transport pass. */
    struct ReflectionBoundarySample
    {
        hase::core::Direction direction;
        double weight = 0.0;
        double wavelength = 0.0;
        bool valid = true;
    };

    /** @brief Selected reflected boundary candidate. */
    struct ReflectionCandidateSample
    {
        std::uint32_t candidateIndex = 0u;
        bool valid = false;
    };

    /**
     * @param rngSeed Seed assigned to one independent statistical batch.
     * @param pass Reflection pass index.
     * @return Deterministic systematic-resampling offset in `(0, 1)`.
     */
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr double reflectionResamplingOffset(
        std::uint32_t const rngSeed,
        std::uint32_t const pass)
    {
        std::uint64_t value = (static_cast<std::uint64_t>(rngSeed) << 32u) | pass;
        value ^= value >> 30u;
        value *= 0xbf58'476d'1ce4'e5b9ull;
        value ^= value >> 27u;
        value *= 0x94d0'49bb'1331'11ebull;
        value ^= value >> 31u;
        std::uint32_t const sample = static_cast<std::uint32_t>(value >> 32u);
        return (static_cast<double>(sample) + 0.5) * (1.0 / 4294967296.0);
    }

    /**
     * @brief Store one ray-owned reflected boundary candidate and terminate the walk.
     * @param mesh Device trace defining the flat face layout.
     * @param rayState Current ray state supplying the boundary position.
     * @param cell Current cell index.
     * @param localFace Hit face within the current cell.
     * @param sample Direction, weight, wavelength, and validity to store.
     * @param candidates Output candidate views.
     * @param candidateIndex Unique ray-owned candidate slot for this pass.
     * @return `BoundaryResult::stop` after handling the boundary.
     */
    ALPAKA_FN_ACC ray::BoundaryResult storeReflectionCandidate(
        hase::data::TraceView const& mesh,
        ray::State auto& rayState,
        std::uint32_t const cell,
        std::uint32_t const localFace,
        ReflectionBoundarySample const sample,
        alpaka::concepts::SpecializationOf<ReflectionCandidateSpans> auto candidates,
        std::uint32_t const candidateIndex)
    {
        if(!sample.valid || sample.weight <= 0.0 || !alpaka::math::isfinite(sample.weight))
            return ray::BoundaryResult::stop;

        candidates.store(
            candidateIndex,
            ReflectionCandidateState{
                rayState.position,
                sample.direction,
                sample.wavelength,
                cell * mesh.numberOfFacesPerCell + localFace},
            sample.weight);
        return ray::BoundaryResult::stop;
    }

    /** @brief Map invalid reflected-candidate weights to zero before building the sampling CDF. */
    struct FilterReflectionSamplingWeight
    {
        [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr double operator()(double const weight) const
        {
            return weight > 0.0 && alpaka::math::isfinite(weight) ? weight : 0.0;
        }
    };

    /** @brief Capture the final unnormalized reflected-candidate prefix. */
    struct CaptureReflectionSamplingTotalWeight
    {
        ALPAKA_FN_HOST_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            std::uint32_t const candidateCount,
            alpaka::concepts::SpecializationOf<ReflectionSamplingSpans> auto sampling) const
        {
            sampling.totalWeight[0u] = candidateCount == 0u ? 0.0 : sampling.cdf[candidateCount - 1u];
        }
    };

    /**
     * @brief Locate one deterministic systematic-resampling target in an ordered CDF.
     * @param cdf Unnormalized non-decreasing cumulative candidate weights.
     * @param candidateCount Number of candidate entries in `cdf`.
     * @param totalWeight Final cumulative weight.
     * @param rayIndex Relaunched ray index within the pass.
     * @param rayCount Total number of relaunched rays.
     * @param offset Seed-derived systematic offset in `(0, 1)`.
     * @return Candidate index whose cumulative interval contains the target.
     */
    template<typename T_Cdf>
    [[nodiscard]] ALPAKA_FN_HOST_ACC std::uint32_t reflectionCandidateIndex(
        T_Cdf const& cdf,
        std::uint32_t const candidateCount,
        double const totalWeight,
        std::uint32_t const rayIndex,
        std::uint32_t const rayCount,
        double const offset)
    {
        double const target = (static_cast<double>(rayIndex) + offset) * (totalWeight / static_cast<double>(rayCount));
        std::uint32_t lower = 0u;
        std::uint32_t upper = candidateCount;
        while(lower < upper)
        {
            std::uint32_t const middle = lower + (upper - lower) / 2u;
            if(cdf[middle] <= target)
                lower = middle + 1u;
            else
                upper = middle;
        }
        return lower < candidateCount ? lower : candidateCount - 1u;
    }

    /**
     * @brief Select one candidate by deterministic systematic weighted resampling.
     * @param candidates Candidate weights defining valid entries.
     * @param sampling Unnormalized ordered cumulative weights and their total.
     * @param candidateCount Number of candidate slots populated for the pass.
     * @param rayIndex Relaunched ray index within the pass.
     * @param rayCount Total number of relaunched rays.
     * @param offset Seed-derived systematic offset in `(0, 1)`.
     * @return Selected candidate, or an invalid sample if the total weight is zero.
     */
    [[nodiscard]] ALPAKA_FN_HOST_ACC ReflectionCandidateSample sampleReflectionCandidate(
        alpaka::concepts::SpecializationOf<ReflectionCandidateSpans> auto candidates,
        alpaka::concepts::SpecializationOf<ReflectionSamplingSpans> auto sampling,
        std::uint32_t const candidateCount,
        std::uint32_t const rayIndex,
        std::uint32_t const rayCount,
        double const offset)
    {
        double const totalWeight = sampling.totalWeight[0u];
        if(candidateCount == 0u || rayCount == 0u || totalWeight <= 0.0)
            return {};

        std::uint32_t const candidateIndex
            = reflectionCandidateIndex(sampling.cdf, candidateCount, totalWeight, rayIndex, rayCount, offset);
        return {candidateIndex, candidates.weights[candidateIndex] > 0.0};
    }
} // namespace hase::kernels::forward
