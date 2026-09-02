/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/HybridBuffer.hpp>
#include <concepts/concepts.hpp>
#include <core/Runtime.hpp>
#include <core/reflectionResampling.hpp>
#include <core/srm.hpp>
#include <core/surfaceReservoir.hpp>
#include <data/TraceData.hpp>
#include <kernels/forward/accumulation.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hase::core
{
    /** @brief Unnormalized forward accumulators and reflected-pass convergence metadata. */
    struct ForwardPhiAseRawResult
    {
        std::vector<double> vertexBatchScoreSum;
        std::vector<unsigned> rseBatchRayCounts;
        std::vector<unsigned> totalRays;
        std::vector<unsigned> droppedRays;
        unsigned rayCount = 0u;
        data::SrmStatus srmStatus = data::SrmStatus::disabled;
        unsigned srmPasses = 0u;
        double srmRemainingFraction = 0.0;
        unsigned srmMaxIterations = 0u;
        unsigned srmDivergenceStreak = 0u;
    };

    /** @brief Customization point for distributing a scalar frame limit over a fixed dimensionality. */
    struct GetScalarDistribution
    {
        template<std::uint32_t T_dim>
        struct Op
        {
            auto operator()(std::unsigned_integral auto const scalarLimit) const
            {
                static_assert(T_dim > 0u);
                using Index = ALPAKA_TYPEOF(scalarLimit);
                using NumFrames = alpaka::Vec<Index, T_dim>;

                Index remainingFactor = scalarLimit;
                auto frameLimit = NumFrames::fill(Index{1u});

                // Put the complete power-of-two factor into the fastest-running extent.
                while(remainingFactor > Index{1u} && remainingFactor % Index{2u} == Index{0u})
                {
                    frameLimit[T_dim - 1u] *= Index{2u};
                    remainingFactor /= Index{2u};
                }

                std::vector<Index> oddPrimeFactors;
                for(Index factor = Index{3u}; factor <= remainingFactor / factor; factor += Index{2u})
                {
                    while(remainingFactor % factor == Index{0u})
                    {
                        oddPrimeFactors.push_back(factor);
                        remainingFactor /= factor;
                    }
                }
                if(remainingFactor > Index{1u})
                    oddPrimeFactors.push_back(remainingFactor);

                if constexpr(T_dim == 1u)
                {
                    for(Index const factor : oddPrimeFactors)
                        frameLimit[0u] *= factor;
                }
                else
                {
                    std::uint32_t slowDimension = T_dim - 2u;
                    for(auto factor = oddPrimeFactors.rbegin(); factor != oddPrimeFactors.rend(); ++factor)
                    {
                        frameLimit[slowDimension] *= *factor;
                        slowDimension = slowDimension == 0u ? T_dim - 2u : slowDimension - 1u;
                    }
                }
                return frameLimit;
            }
        };
    };

    /**
     * @brief Distribute a scalar frame limit over a requested dimensionality.
     * @param scalarLimit Maximum product of the returned frame extents.
     * @param anyWithDim Object selecting the dimensionality of the returned vector.
     * @return Per-dimension frame limits selected by the matching customization.
     */
    auto getScalarDistribution(
        std::unsigned_integral auto const scalarLimit,
        alpaka::concepts::HasStaticDim auto const& anyWithDim)
    {
        constexpr auto dim = alpaka::getDim(ALPAKA_TYPEOF(anyWithDim){});
        return GetScalarDistribution::Op<dim>{}(scalarLimit);
    }

    /**
     * @brief Construct a one-dimensional ray frame specification with 128 threads per frame.
     * @param rayCount Number of rays covered by the frame specification.
     * @param queue Queue whose device limits the number of frames to four full waves.
     * @return Ray frame specification clamped to the queue's device.
     */
    auto getRayFrameSpec(unsigned const rayCount, hase::concepts::Queue auto const& queue)
    {
        auto const numMultiprocessors = queue.getDevice().getDeviceProperties().multiProcessorCount;
        auto const numFrames = getScalarDistribution(alpaka::divCeil(rayCount, 128u), alpaka::Vec{1u});
        // auto const numClamped = getScalarDistribution(4u * numMultiprocessors, alpaka::Vec{1u});
        auto const numClamped = getScalarDistribution(8u * numMultiprocessors, alpaka::Vec{1u});
        return alpaka::onHost::FrameSpec{numFrames.min(numClamped), alpaka::Vec{128u}};
    }

    /**
     * @brief Trace direct and recursively reflected forward histories with weighted resampling.
     * @tparam T_Device Device type owned by `devBundle` and `scratch`.
     * @tparam T_Exec Executor used for all kernel launches.
     * @param devBundle Device and executor pair used to construct frame specifications.
     * @param queue Queue on the same device; this function waits at pass boundaries.
     * @param mesh Non-owning device view of geometry, materials, and excitation.
     * @param experiment Trace and reflection configuration.
     * @param result Raw result whose SRM status fields are updated.
     * @param rayCount Number of primary histories.
     * @param rseBatch Statistical batch receiving the accumulated scores.
     * @param sourceStrengthTotal Total source strength used to normalize primary histories.
     * @param vertexBatchScoreSum Per-batch material-vertex score accumulator.
     * @param volumeRayVisits Per-cell visit counter.
     * @param droppedRays Per-cell counter for histories dropped after traversal failures.
     * @param threadLocalStridingRNG Seed for primary and reflected sampling.
     * @param srmControls Reflection pass convergence and divergence limits.
     * @param scratch Persistent reflected-candidate and sampling storage.
     */
    template<alpaka::onHost::concepts::Device T_Device, alpaka::concepts::Executor T_Exec>
    void runForwardSrm(
        alpakaUtils::DevBundle<T_Device, T_Exec>& devBundle,
        concepts::Queue auto const& queue,
        data::TraceView const mesh,
        AseTraceControls const& experiment,
        ForwardPhiAseRawResult& result,
        unsigned const rayCount,
        unsigned const rseBatch,
        double const sourceStrengthTotal,
        alpaka::concepts::IBuffer<double> auto& vertexBatchScoreSum,
        alpaka::concepts::IBuffer<std::uint32_t> auto& volumeRayVisits,
        alpaka::concepts::IBuffer<std::uint32_t> auto& droppedRays,
        unsigned const threadLocalStridingRNG,
        SrmControls const srmControls,
        ReflectionResamplingScratch<T_Device>& scratch,
        hase::kernels::forward::concepts::TracePolicy auto const diagnostics)
    {
        auto accumulationSpans = hase::kernels::forward::ForwardAccumulationSpans{
            vertexBatchScoreSum.getMdSpan(),
            volumeRayVisits.getMdSpan(),
            droppedRays.getMdSpan()};
        scratch.validate(rayCount);
        scratch.clear(queue, scratch.first);
        scratch.clear(queue, scratch.second);
        alpaka::onHost::wait(queue);
        auto const rayFrameSpec = getRayFrameSpec(rayCount, queue);
        queue.enqueue(
            rayFrameSpec,
            alpaka::KernelBundle{
                hase::kernels::forward::AccumulateForwardPhiAse{},
                hase::kernels::forward::TracePolicyList{
                    hase::kernels::forward::tracePolicy::source::volume,
                    hase::kernels::forward::tracePolicy::cell::forwardAse,
                    hase::kernels::forward::tracePolicy::boundary::reflectionCandidates,
                    hase::kernels::forward::tracePolicy::position::exact,
                    diagnostics},
                mesh,
                rayCount,
                rseBatch,
                sourceStrengthTotal,
                accumulationSpans,
                scratch.first.view(),
                threadLocalStridingRNG});
        alpaka::onHost::wait(queue);

        bool inputA = true;
        double const initialWeight = scratch.updateSampling(devBundle, queue, scratch.first, rayCount);
        if(initialWeight == 0.0)
        {
            result.srmStatus = data::SrmStatus::converged;
            return;
        }

        double previousWeight = initialWeight;
        unsigned growCount = 0u;
        result.srmStatus = data::SrmStatus::maxIterations;
        result.srmRemainingFraction = 1.0;
        for(unsigned pass = 1u; pass <= srmControls.maxIterations; ++pass)
        {
            double const sourceWeight = previousWeight / static_cast<double>(rayCount);
            // Alternate exact candidate banks between reflected passes.
            if(inputA)
            {
                scratch.clear(queue, scratch.second);
                alpaka::onHost::wait(queue);
                queue.enqueue(
                    rayFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AccumulateResampledReflectedPhiAse{},
                        hase::kernels::forward::TracePolicyList{
                            hase::kernels::forward::tracePolicy::source::reflectionCandidates,
                            hase::kernels::forward::tracePolicy::cell::forwardAse,
                            hase::kernels::forward::tracePolicy::boundary::reflectionCandidates,
                            hase::kernels::forward::tracePolicy::position::exact,
                            diagnostics},
                        mesh,
                        rayCount,
                        rseBatch,
                        sourceWeight,
                        accumulationSpans,
                        scratch.first.view(),
                        scratch.samplingView(),
                        scratch.second.view(),
                        threadLocalStridingRNG,
                        pass});
            }
            else
            {
                scratch.clear(queue, scratch.first);
                alpaka::onHost::wait(queue);
                queue.enqueue(
                    rayFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AccumulateResampledReflectedPhiAse{},
                        hase::kernels::forward::TracePolicyList{
                            hase::kernels::forward::tracePolicy::source::reflectionCandidates,
                            hase::kernels::forward::tracePolicy::cell::forwardAse,
                            hase::kernels::forward::tracePolicy::boundary::reflectionCandidates,
                            hase::kernels::forward::tracePolicy::position::exact,
                            diagnostics},
                        mesh,
                        rayCount,
                        rseBatch,
                        sourceWeight,
                        accumulationSpans,
                        scratch.second.view(),
                        scratch.samplingView(),
                        scratch.first.view(),
                        threadLocalStridingRNG,
                        pass});
            }
            alpaka::onHost::wait(queue);
            inputA = !inputA;

            double const currentWeight
                = scratch.updateSampling(devBundle, queue, inputA ? scratch.first : scratch.second, rayCount);
            result.srmPasses = pass;
            result.srmRemainingFraction = currentWeight / initialWeight;
            if(currentWeight > previousWeight)
            {
                ++growCount;
                if(growCount >= srmControls.divergenceStreak)
                {
                    result.srmStatus = data::SrmStatus::diverged;
                    break;
                }
            }
            else
            {
                growCount = 0u;
                if(alpaka::math::abs(currentWeight - previousWeight) / alpaka::math::max(currentWeight, 1.0e-30)
                   < experiment.reflectionTolerance)
                {
                    result.srmStatus = data::SrmStatus::stable;
                    break;
                }
            }
            if(result.srmRemainingFraction < experiment.reflectionTolerance)
            {
                result.srmStatus = data::SrmStatus::converged;
                break;
            }
            previousWeight = currentWeight;
        }
    }

    /** @brief Trace forward histories through a bounded face-indexed surface reservoir. */
    template<
        alpaka::onHost::concepts::Device T_Device,
        alpaka::concepts::Executor T_Exec,
        hase::kernels::forward::SurfaceReservoirPositionPolicy T_PositionPolicy>
    void runForwardSurfaceSrm(
        alpakaUtils::DevBundle<T_Device, T_Exec>& devBundle,
        concepts::Queue auto const& queue,
        hase::data::TraceView const mesh,
        AseTraceControls const& experiment,
        ForwardPhiAseRawResult& result,
        std::uint32_t const rayCount,
        std::uint32_t const rseBatch,
        double const sourceStrengthTotal,
        alpaka::concepts::IBuffer<double> auto& vertexBatchScoreSum,
        alpaka::concepts::IBuffer<std::uint32_t> auto& volumeRayVisits,
        alpaka::concepts::IBuffer<std::uint32_t> auto& droppedRays,
        std::uint32_t const rngSeed,
        SrmControls const srmControls,
        SurfaceReservoirScratch<T_Device, T_PositionPolicy>& scratch,
        hase::kernels::forward::concepts::TracePolicy auto const diagnostics)
    {
        constexpr auto positionPolicy = []
        {
            if constexpr(std::same_as<T_PositionPolicy, hase::kernels::forward::surfaceReservoirPosition::Exact>)
                return hase::kernels::forward::tracePolicy::position::exact;
            else
                return hase::kernels::forward::tracePolicy::position::centroid;
        }();
        auto accumulation = hase::kernels::forward::ForwardAccumulationSpans{
            vertexBatchScoreSum.getMdSpan(),
            volumeRayVisits.getMdSpan(),
            droppedRays.getMdSpan()};
        std::uint32_t const faceCount = mesh.numberOfCells * mesh.numberOfFacesPerCell;
        auto const slotsPerFace = alpaka::Vec{experiment.surfaceReservoirSize};
        scratch.validate(faceCount, rayCount);
        scratch.clear(queue, scratch.reservoir.first);
        scratch.clear(queue, scratch.reservoir.second);
        alpaka::onHost::wait(queue);
        auto const rayFrameSpec = getRayFrameSpec(rayCount, queue);
        queue.enqueue(
            rayFrameSpec,
            alpaka::KernelBundle{
                hase::kernels::forward::AccumulateForwardPhiAseSurfaceReservoir{},
                hase::kernels::forward::TracePolicyList{
                    hase::kernels::forward::tracePolicy::source::volume,
                    hase::kernels::forward::tracePolicy::cell::forwardAse,
                    hase::kernels::forward::tracePolicy::boundary::surfaceReservoir,
                    positionPolicy,
                    diagnostics},
                mesh,
                rayCount,
                rseBatch,
                sourceStrengthTotal,
                accumulation,
                scratch.reservoir.first.view(slotsPerFace),
                rngSeed});
        alpaka::onHost::wait(queue);

        bool inputFirst = true;
        double const initialWeight
            = scratch.updateSampling(devBundle, queue, scratch.reservoir.first, slotsPerFace, rayCount, rngSeed, 0u);
        if(initialWeight == 0.0)
        {
            result.srmStatus = data::SrmStatus::converged;
            return;
        }

        double previousWeight = initialWeight;
        std::uint32_t growCount = 0u;
        result.srmStatus = data::SrmStatus::maxIterations;
        result.srmRemainingFraction = 1.0;
        for(std::uint32_t pass = 1u; pass <= srmControls.maxIterations; ++pass)
        {
            double const sourceWeight = previousWeight / static_cast<double>(rayCount);
            auto& input = inputFirst ? scratch.reservoir.first : scratch.reservoir.second;
            auto& output = inputFirst ? scratch.reservoir.second : scratch.reservoir.first;
            scratch.clear(queue, output);
            alpaka::onHost::wait(queue);
            queue.enqueue(
                rayFrameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::AccumulateSurfaceReservoirForwardPhiAse{},
                    hase::kernels::forward::TracePolicyList{
                        hase::kernels::forward::tracePolicy::source::surfaceReservoir,
                        hase::kernels::forward::tracePolicy::cell::forwardAse,
                        hase::kernels::forward::tracePolicy::boundary::surfaceReservoir,
                        positionPolicy,
                        diagnostics},
                    mesh,
                    rayCount,
                    rseBatch,
                    sourceWeight,
                    accumulation,
                    input.view(slotsPerFace),
                    scratch.samplingView(faceCount <= rayCount),
                    output.view(slotsPerFace),
                    rngSeed,
                    pass});
            alpaka::onHost::wait(queue);
            inputFirst = !inputFirst;

            double const currentWeight
                = scratch.updateSampling(devBundle, queue, output, slotsPerFace, rayCount, rngSeed, pass);
            result.srmPasses = pass;
            result.srmRemainingFraction = currentWeight / initialWeight;
            if(currentWeight > previousWeight)
            {
                ++growCount;
                if(growCount >= srmControls.divergenceStreak)
                {
                    result.srmStatus = data::SrmStatus::diverged;
                    break;
                }
            }
            else
            {
                growCount = 0u;
                if(alpaka::math::abs(currentWeight - previousWeight) / alpaka::math::max(currentWeight, 1.0e-30)
                   < experiment.reflectionTolerance)
                {
                    result.srmStatus = data::SrmStatus::stable;
                    break;
                }
            }
            if(result.srmRemainingFraction < experiment.reflectionTolerance)
            {
                result.srmStatus = data::SrmStatus::converged;
                break;
            }
            previousWeight = currentWeight;
        }
    }
} // namespace hase::core
