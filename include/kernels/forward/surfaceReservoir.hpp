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
#include <kernels/forward/randomHistory.hpp>
#include <random/randomEngine.hpp>

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace hase::kernels::forward
{
    namespace surfaceReservoirPosition
    {
        struct Policy
        {
        };

        struct Exact : Policy
        {
        };

        struct Centroid : Policy
        {
        };

        inline constexpr Exact exact;
        inline constexpr Centroid centroid;
    } // namespace surfaceReservoirPosition

    template<typename T>
    concept SurfaceReservoirPositionPolicy
        = std::derived_from<std::remove_cvref_t<T>, surfaceReservoirPosition::Policy>;

    template<alpaka::concepts::IView<double> T_PositionView>
    struct ExactSurfaceReservoirPositionSpans
    {
        hase::core::PositionViewSoA<T_PositionView> positions;
        hase::core::PositionViewSoA<T_PositionView> candidatePositions;
    };

    struct CentroidSurfaceReservoirPositionSpans
    {
    };

    /** @brief Kernel views of one face-indexed weighted reservoir bank. */
    template<
        alpaka::concepts::IView<std::uint32_t> T_Counts,
        alpaka::concepts::KernelArg T_PositionSpans,
        alpaka::concepts::IView<double> T_DirectionView,
        alpaka::concepts::IView<double> T_Weights,
        alpaka::concepts::IView<double> T_Wavelengths,
        alpaka::concepts::IView<double> T_FaceWeights,
        alpaka::concepts::IView<std::uint64_t> T_SelectionKeys,
        alpaka::concepts::Vector T_SlotsPerFace>
    struct SurfaceReservoirSpans
    {
        T_Counts counts;
        T_PositionSpans positionSpans;
        hase::core::DirectionViewSoA<T_DirectionView> directions;
        T_Weights weights;
        T_Wavelengths wavelengths;
        T_FaceWeights faceWeights;
        T_SelectionKeys selectionKeys;
        hase::core::DirectionViewSoA<T_DirectionView> candidateDirections;
        T_Weights candidateWeights;
        T_Wavelengths candidateWavelengths;
        T_SlotsPerFace slotsPerFace;
    };

    template<alpaka::concepts::IView<double> T_PositionView>
    ALPAKA_FN_ACC inline void captureSurfaceReservoirPosition(
        ExactSurfaceReservoirPositionSpans<T_PositionView> positionSpans,
        std::uint32_t const candidateIndex,
        hase::core::Position const position)
    {
        positionSpans.candidatePositions.x[candidateIndex] = position.x;
        positionSpans.candidatePositions.y[candidateIndex] = position.y;
        positionSpans.candidatePositions.z[candidateIndex] = position.z;
    }

    ALPAKA_FN_ACC inline void captureSurfaceReservoirPosition(
        CentroidSurfaceReservoirPositionSpans,
        std::uint32_t,
        hase::core::Position)
    {
    }

    template<alpaka::concepts::IView<double> T_PositionView>
    ALPAKA_FN_ACC inline void finalizeSurfaceReservoirPosition(
        ExactSurfaceReservoirPositionSpans<T_PositionView> positionSpans,
        std::uint32_t const destination,
        std::uint32_t const candidateIndex)
    {
        positionSpans.positions.x[destination] = positionSpans.candidatePositions.x[candidateIndex];
        positionSpans.positions.y[destination] = positionSpans.candidatePositions.y[candidateIndex];
        positionSpans.positions.z[destination] = positionSpans.candidatePositions.z[candidateIndex];
    }

    ALPAKA_FN_ACC inline void finalizeSurfaceReservoirPosition(
        CentroidSurfaceReservoirPositionSpans,
        std::uint32_t,
        std::uint32_t)
    {
    }

    template<alpaka::concepts::IView<double> T_PositionView>
    [[nodiscard]] ALPAKA_FN_ACC inline hase::core::Position restoreSurfaceReservoirPosition(
        ExactSurfaceReservoirPositionSpans<T_PositionView> positionSpans,
        hase::data::TraceView const&,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t const slotIndex)
    {
        return positionSpans.positions.at(slotIndex);
    }

    [[nodiscard]] ALPAKA_FN_ACC inline hase::core::Position restoreSurfaceReservoirPosition(
        CentroidSurfaceReservoirPositionSpans,
        hase::data::TraceView const& mesh,
        std::uint32_t const cell,
        std::uint32_t const localFace,
        std::uint32_t)
    {
        return ray::restoreSrmPosition(ray::srmPosition::centroid, mesh, cell, localFace);
    }

    /** @brief Kernel views of the normalized face CDF and ray-face assignments. */
    template<
        alpaka::concepts::IView<double> T_Cdf,
        alpaka::concepts::IView<double> T_TotalWeight,
        alpaka::concepts::IView<std::uint32_t> T_RayFaces>
    struct SurfaceReservoirSamplingCdfSpans
    {
        T_Cdf cdf;
        T_TotalWeight totalWeight;
        T_RayFaces rayFaces;
        bool useFaceStratification;
    };

    /** @brief Selected reservoir slot and its owning face. */
    struct SurfaceReservoirSample
    {
        std::uint32_t faceId = 0u;
        std::uint32_t slotIndex = 0u;
        bool valid = false;
    };

    /** @brief Reflected boundary state offered to reservoir sampling. */
    struct SurfaceReservoirBoundarySample
    {
        hase::core::Direction direction;
        double weight = 0.0;
        double wavelength = 0.0;
        bool valid = true;
    };

    /**
     * @brief Add one weighted boundary state using bounded reservoir replacement.
     * @param acc Accelerator context used for atomic accumulation.
     * @param faceId Flat local face index.
     * @param position Boundary-hit position.
     * @param direction Relaunch direction.
     * @param weight Statistical sample weight.
     * @param wavelength Ray wavelength in metres.
     * @param reservoir Output reservoir views.
     * @param candidateIndex Unique ray-owned candidate slot for this pass.
     * @param rng Random engine used to derive the weighted priority.
     */
    ALPAKA_FN_ACC void depositSurfaceReservoirSample(
        alpaka::onAcc::concepts::Acc auto const& acc,
        std::uint32_t const faceId,
        hase::core::Position const position,
        hase::core::Direction const direction,
        double const weight,
        double const wavelength,
        alpaka::concepts::SpecializationOf<SurfaceReservoirSpans> auto reservoir,
        std::uint32_t const candidateIndex,
        alpaka::rand::concepts::UniformRandomEngine auto& rng)
    {
        alpaka::concepts::Vector auto slotsPerFace = reservoir.slotsPerFace;
        if(weight <= 0.0 || !alpaka::math::isfinite(weight))
            return;

        captureSurfaceReservoirPosition(reservoir.positionSpans, candidateIndex, position);
        reservoir.candidateDirections.x[candidateIndex] = direction.x;
        reservoir.candidateDirections.y[candidateIndex] = direction.y;
        reservoir.candidateDirections.z[candidateIndex] = direction.z;
        reservoir.candidateWeights[candidateIndex] = weight;
        reservoir.candidateWavelengths[candidateIndex] = wavelength;

        alpaka::onAcc::atomicAdd(acc, &reservoir.counts[faceId], 1u);
        alpaka::onAcc::atomicAdd(acc, &reservoir.faceWeights[faceId], weight);

        double const uniform = alpaka::rand::distribution::UniformReal<double, alpaka::rand::interval::OO>{}(rng);
        double const priority = -alpaka::math::log(uniform) / weight;
        float const boundedPriority
            = static_cast<float>(alpaka::math::min(priority, static_cast<double>(std::numeric_limits<float>::max())));
        std::uint64_t candidateKey
            = (static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(boundedPriority)) << 32u) | candidateIndex;
        constexpr std::uint64_t emptyKey = std::numeric_limits<std::uint64_t>::max();
        std::uint32_t const offset = faceId * slotsPerFace.x();
        for(std::uint32_t slot = 0u; slot < slotsPerFace.x(); ++slot)
        {
            std::uint64_t const displaced
                = alpaka::onAcc::atomicMin(acc, &reservoir.selectionKeys[offset + slot], candidateKey);
            if(displaced == emptyKey)
                break;
            if(displaced > candidateKey)
                candidateKey = displaced;
        }
    }

    /**
     * @brief Store a valid reflected boundary sample and terminate the current walk.
     * @param acc Accelerator context used for atomic reservoir updates.
     * @param mesh Device trace defining the flat face layout.
     * @param rayState Current ray state supplying the boundary position.
     * @param cell Current cell index.
     * @param localFace Hit face within the current cell.
     * @param sample Direction, weight, wavelength, and validity to store.
     * @param reservoir Output reservoir views.
     * @param candidateIndex Unique ray-owned candidate slot for this pass.
     * @param rng Random engine used for reservoir replacement.
     * @return `BoundaryResult::stop` after handling the boundary.
     */
    ALPAKA_FN_ACC ray::BoundaryResult storeSurfaceReservoirBoundarySample(
        alpaka::onAcc::concepts::Acc auto const& acc,
        hase::data::TraceView const& mesh,
        ray::State auto& rayState,
        std::uint32_t const cell,
        std::uint32_t const localFace,
        SurfaceReservoirBoundarySample const sample,
        alpaka::concepts::SpecializationOf<SurfaceReservoirSpans> auto reservoir,
        std::uint32_t const candidateIndex,
        alpaka::rand::concepts::UniformRandomEngine auto& rng)
    {
        if(!sample.valid)
            return ray::BoundaryResult::stop;

        depositSurfaceReservoirSample(
            acc,
            cell * mesh.numberOfFacesPerCell + localFace,
            rayState.position,
            sample.direction,
            sample.weight,
            sample.wavelength,
            reservoir,
            candidateIndex,
            rng);
        return ray::BoundaryResult::stop;
    }

    /**
     * @brief Select a face and weighted slot for one relaunched history.
     * @param reservoir Input reservoir views.
     * @param sampling Normalized face CDF and optional stratified assignments.
     * @param faceCount Number of local cell faces.
     * @param rayIndex Ray index used by a precomputed stratified assignment.
     * @param rng Random engine used for unstratified face and slot selection.
     * @return Valid selected slot, or an invalid sample when no positive weight exists.
     */
    [[nodiscard]] ALPAKA_FN_ACC SurfaceReservoirSample sampleSurfaceReservoir(
        alpaka::concepts::SpecializationOf<SurfaceReservoirSpans> auto reservoir,
        alpaka::concepts::SpecializationOf<SurfaceReservoirSamplingCdfSpans> auto sampling,
        std::uint32_t const faceCount,
        std::uint32_t const rayIndex,
        alpaka::rand::concepts::UniformRandomEngine auto& rng)
    {
        alpaka::concepts::Vector auto slotsPerFace = reservoir.slotsPerFace;
        if(faceCount == 0u || sampling.totalWeight[0u] <= 0.0)
            return {};

        std::uint32_t faceId = sampling.useFaceStratification ? sampling.rayFaces[rayIndex] : 0u;
        if(!sampling.useFaceStratification)
        {
            double const target = alpaka::rand::distribution::UniformReal<double>{}(rng);
            std::uint32_t lower = 0u;
            std::uint32_t upper = faceCount;
            while(lower < upper)
            {
                std::uint32_t const middle = lower + (upper - lower) / 2u;
                if(sampling.cdf[middle] <= target)
                    lower = middle + 1u;
                else
                    upper = middle;
            }
            faceId = lower < faceCount ? lower : faceCount - 1u;
        }

        std::uint32_t const filledSlots = alpaka::math::min(reservoir.counts[faceId], slotsPerFace.x());
        if(filledSlots == 0u)
            return {};
        std::uint32_t const offset = faceId * slotsPerFace.x();
        std::uint32_t localSlot = 0u;
        if(sampling.useFaceStratification)
            localSlot = rayIndex % filledSlots;
        else
        {
            localSlot = static_cast<std::uint32_t>(
                alpaka::rand::distribution::UniformReal<double>{}(rng) * static_cast<double>(filledSlots));
            if(localSlot >= filledSlots)
                localSlot = filledSlots - 1u;
        }
        return {faceId, offset + localSlot, true};
    }

    /** @brief Gather deterministic top-K priority winners into the sampled reservoir slots. */
    struct FinalizeSurfaceReservoir
    {
        ALPAKA_FN_HOST_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t const faceCount,
            alpaka::concepts::SpecializationOf<SurfaceReservoirSpans> auto reservoir) const
        {
            constexpr std::uint64_t emptyKey = std::numeric_limits<std::uint64_t>::max();
            alpaka::concepts::Vector auto slotsPerFace = reservoir.slotsPerFace;
            for(auto [face] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{faceCount}))
            {
                std::uint32_t const offset = face * slotsPerFace.x();
                std::uint32_t filledSlots = 0u;
                for(std::uint32_t slot = 0u; slot < slotsPerFace.x(); ++slot)
                {
                    std::uint64_t const key = reservoir.selectionKeys[offset + slot];
                    if(key == emptyKey)
                        break;
                    std::uint32_t const candidateIndex = static_cast<std::uint32_t>(key);
                    std::uint32_t const destination = offset + slot;
                    finalizeSurfaceReservoirPosition(reservoir.positionSpans, destination, candidateIndex);
                    reservoir.directions.x[destination] = reservoir.candidateDirections.x[candidateIndex];
                    reservoir.directions.y[destination] = reservoir.candidateDirections.y[candidateIndex];
                    reservoir.directions.z[destination] = reservoir.candidateDirections.z[candidateIndex];
                    reservoir.weights[destination] = reservoir.candidateWeights[candidateIndex];
                    reservoir.wavelengths[destination] = reservoir.candidateWavelengths[candidateIndex];
                    ++filledSlots;
                }
                reservoir.counts[face] = filledSlots;
            }
        }
    };

    /** @brief Device operation normalizing cumulative face weights to `[0, 1]`. */
    struct NormalizeSurfaceReservoirSamplingCdf
    {
        ALPAKA_FN_HOST_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t const faceCount,
            alpaka::concepts::SpecializationOf<SurfaceReservoirSamplingCdfSpans> auto samplingCdf) const
        {
            double const totalWeight = samplingCdf.totalWeight[0u];
            for(auto [face] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{faceCount}))
                samplingCdf.cdf[face] = totalWeight > 0.0 ? samplingCdf.cdf[face] / totalWeight : 0.0;
        }
    };

    /** @brief Device operation capturing the final unnormalized face prefix. */
    struct CaptureSurfaceReservoirSamplingTotalWeight
    {
        ALPAKA_FN_HOST_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            std::uint32_t const faceCount,
            alpaka::concepts::SpecializationOf<SurfaceReservoirSamplingCdfSpans> auto samplingCdf) const
        {
            samplingCdf.totalWeight[0u] = faceCount == 0u ? 0.0 : samplingCdf.cdf[faceCount - 1u];
        }
    };

    /** @brief Device operation generating one pass-specific systematic offset. */
    struct GenerateSurfaceReservoirSystematicOffset
    {
        ALPAKA_FN_HOST_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            alpaka::concepts::IView<double> auto systematicOffset,
            std::uint32_t const rngSeed,
            std::uint32_t const pass) const
        {
            auto rng = hase::random::makeRandomEngine(rngSeed, surfaceSamplingHistoryId(pass));
            systematicOffset[0u] = alpaka::rand::distribution::UniformReal<double, alpaka::rand::interval::OO>{}(rng);
        }
    };

    /** @brief Device operation assigning systematic sample counts to faces. */
    struct AssignSurfaceReservoirStratifiedRayCounts
    {
        ALPAKA_FN_HOST_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t const faceCount,
            std::uint32_t const rayCount,
            alpaka::concepts::SpecializationOf<SurfaceReservoirSamplingCdfSpans> auto samplingCdf,
            alpaka::concepts::IView<double> auto systematicOffset,
            alpaka::concepts::IView<std::uint32_t> auto rayCounts) const
        {
            double const offset = systematicOffset[0u];
            for(auto [face] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{faceCount}))
            {
                double const lowerCdf = face == 0u ? 0.0 : samplingCdf.cdf[face - 1u];
                double const scaledLower = static_cast<double>(rayCount) * lowerCdf - offset;
                double const scaledUpper = static_cast<double>(rayCount) * samplingCdf.cdf[face] - offset;
                rayCounts[face]
                    = static_cast<std::uint32_t>(alpaka::math::floor(scaledUpper) - alpaka::math::floor(scaledLower));
            }
        }
    };

    /** @brief Device operation marking face changes before an inclusive fill scan. */
    struct MarkSurfaceReservoirStratifiedFaceStarts
    {
        ALPAKA_FN_HOST_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            std::uint32_t const faceCount,
            std::uint32_t const rayCount,
            alpaka::concepts::IView<std::uint32_t> auto rayOffsets,
            alpaka::concepts::IView<std::uint32_t> auto rayFaces) const
        {
            for(auto [face] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{1u, faceCount}))
            {
                std::uint32_t const firstRay = rayOffsets[face];
                if(firstRay < rayCount)
                    alpaka::onAcc::atomicAdd(acc, &rayFaces[firstRay], 1u);
            }
        }
    };
} // namespace hase::kernels::forward
