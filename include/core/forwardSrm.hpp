/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpakaUtils/DevBundle.hpp>
#include <concepts/concepts.hpp>
#include <core/Runtime.hpp>
#include <core/reflectionTail.hpp>
#include <core/srm.hpp>
#include <core/surfaceReservoir.hpp>
#include <data/TraceData.hpp>
#include <kernels/forward/accumulation.hpp>
#include <kernels/reflectionTail.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hase::core
{
    /** @brief Unnormalized forward accumulators and boundary-pass convergence metadata. */
    struct ForwardPhiAseRawResult
    {
        std::vector<double> vertexBatchScoreSum;
        std::vector<unsigned> rseBatchRayCounts;
        std::vector<unsigned> totalRays;
        std::vector<unsigned> droppedRays;
        unsigned rayCount = 0u;
        data::BoundaryStatus boundaryStatus = data::BoundaryStatus::disabled;
        unsigned boundaryPasses = 0u;
        double boundaryRemainingFraction = 0.0;
        unsigned boundaryMaxPasses = 0u;
        unsigned boundaryDivergenceStreak = 0u;
        double boundaryGamma = 0.0;
        double boundaryGammaStandardError = 0.0;
        double boundaryTailFactor = 0.0;
        double boundaryTailClosure = 0.0;
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
     * @brief Construct a one-dimensional ray frame specification.
     * @param rayCount Number of rays covered by the frame specification.
     * @param queue Queue whose device limits the number of frames to eight full waves.
     * @param threadsPerFrame Number of worker threads in each frame.
     * @return Ray frame specification clamped to the queue's device.
     */
    auto getRayFrameSpec(
        unsigned const rayCount,
        hase::concepts::Queue auto const& queue,
        unsigned const threadsPerFrame = 128u)
    {
        auto const numMultiprocessors = queue.getDevice().getDeviceProperties().multiProcessorCount;
        auto const numFrames = getScalarDistribution(alpaka::divCeil(rayCount, threadsPerFrame), alpaka::Vec{1u});
        // auto const numClamped = getScalarDistribution(4u * numMultiprocessors, alpaka::Vec{1u});
        auto const numClamped = getScalarDistribution(8u * numMultiprocessors, alpaka::Vec{1u});
        return alpaka::onHost::FrameSpec{numFrames.min(numClamped), alpaka::Vec{threadsPerFrame}};
    }

    /** @brief Select the forward-trace block size for the compile-time diagnostics path. */
    ALPAKA_FN_HOST_ACC constexpr unsigned forwardRayThreadsPerFrame(
        hase::kernels::forward::tracePolicy::diagnostics::None)
    {
        // The performance specialization carries no dropped-ray failure state and uses
        // the wider launch selected for throughput.
        return 512u;
    }

    /** @brief Keep diagnostic forward traces within their larger resource envelope. */
    ALPAKA_FN_HOST_ACC constexpr unsigned forwardRayThreadsPerFrame(
        hase::kernels::forward::tracePolicy::diagnostics::Enabled)
    {
        // Visit and dropped-ray accounting add atomics and state, so diagnostics use
        // the conservative launch shape intended for safe validation runs.
        return 128u;
    }

    /** @brief Trace domain-aware forward histories through a bounded per-face surface reservoir. */
    template<
        alpaka::onHost::concepts::Device T_Device,
        alpaka::concepts::Executor T_Exec,
        hase::kernels::forward::SurfaceReservoirPositionPolicy T_PositionPolicy>
    void runForwardSrm(
        alpakaUtils::DevBundle<T_Device, T_Exec>& devBundle,
        concepts::Queue auto const& queue,
        hase::data::TraceView const mesh,
        AseTraceControls const& experiment,
        ForwardPhiAseRawResult& result,
        std::uint32_t const rayCount,
        std::uint32_t const rseBatch,
        double const sourceStrengthTotal,
        alpaka::concepts::IBuffer<double> auto& vertexBatchScoreSum,
        alpaka::concepts::IBuffer<double> auto& boundaryTailSnapshot,
        alpaka::concepts::IBuffer<std::uint32_t> auto& volumeRayVisits,
        alpaka::concepts::IBuffer<std::uint32_t> auto& droppedRays,
        std::uint32_t const rngSeed,
        SrmControls const srmControls,
        SurfaceReservoirScratch<T_Device, T_PositionPolicy>& scratch,
        hase::kernels::forward::concepts::TracePolicy auto const diagnostics,
        hase::data::AseDomainInterfaceView const interfaceMap = {},
        hase::data::AseDomainSourceView const domainSources = {},
        std::span<std::uint32_t const> const domainRayCounts = {},
        std::span<double const> const domainSourceWeights = {},
        std::span<std::uint32_t const> const domainPopulationCounts = {})
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
        std::uint32_t populationRayCount = rayCount;
        if(!domainPopulationCounts.empty())
        {
            populationRayCount = 0u;
            for(auto const count : domainPopulationCounts)
                populationRayCount += count;
            if(populationRayCount == 0u && rayCount != 0u)
                throw std::invalid_argument("a non-empty SRM launch requires a relaunch population");
        }
        scratch.validate(faceCount, std::max(rayCount, populationRayCount));
        scratch.clear(queue, scratch.reservoir.first);
        scratch.clear(queue, scratch.reservoir.second);
        auto updateSampling = [&](auto& bank, std::uint32_t const pass)
        {
            if(domainPopulationCounts.empty())
                return scratch.updateSampling(devBundle, queue, bank, slotsPerFace, populationRayCount, rngSeed, pass);
            return scratch.updateDomainSampling(
                devBundle,
                queue,
                bank,
                slotsPerFace,
                interfaceMap,
                domainPopulationCounts,
                mesh.numberOfFacesPerCell,
                rngSeed,
                pass);
        };

        auto const rayFrameSpec = getRayFrameSpec(rayCount, queue);
        if(domainRayCounts.empty())
            queue.enqueue(
                rayFrameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::AccumulateForwardPhiAseReservoir{},
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
                    rngSeed,
                    interfaceMap,
                    experiment.useReflections});
        else
        {
            std::uint32_t candidateOffset = 0u;
            for(std::uint32_t domain = 0u; domain < domainRayCounts.size(); ++domain)
            {
                auto const count = domainRayCounts[domain];
                if(count == 0u)
                    continue;
                auto const domainFrame = getRayFrameSpec(count, queue);
                queue.enqueue(
                    domainFrame,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AccumulateDomainForwardPhiAseReservoir{},
                        hase::kernels::forward::TracePolicyList{
                            hase::kernels::forward::tracePolicy::source::volume,
                            hase::kernels::forward::tracePolicy::cell::forwardAse,
                            hase::kernels::forward::tracePolicy::boundary::surfaceReservoir,
                            positionPolicy,
                            diagnostics},
                        mesh,
                        domainSources,
                        domain,
                        count,
                        candidateOffset,
                        rseBatch,
                        domainSourceWeights[domain],
                        accumulation,
                        scratch.reservoir.first.view(slotsPerFace),
                        rngSeed,
                        interfaceMap,
                        experiment.useReflections});
                candidateOffset += count;
            }
            if(candidateOffset != rayCount)
                throw std::runtime_error("domain batch ray counts do not match the SRM launch total");
        }

        bool inputFirst = true;
        double const initialWeight = updateSampling(scratch.reservoir.first, 0u);
        if(initialWeight == 0.0)
        {
            result.boundaryStatus = data::BoundaryStatus::converged;
            return;
        }

        double previousWeight = initialWeight;
        std::vector<double> residualWeightFractions{1.0};
        std::uint32_t growCount = 0u;
        result.boundaryStatus = data::BoundaryStatus::maxPasses;
        result.boundaryRemainingFraction = 1.0;
        for(std::uint32_t pass = 1u; pass <= srmControls.maxIterations; ++pass)
        {
            double const sourceWeight = previousWeight / static_cast<double>(populationRayCount);
            auto const relaunchFrameSpec = getRayFrameSpec(populationRayCount, queue);
            auto& input = inputFirst ? scratch.reservoir.first : scratch.reservoir.second;
            auto& output = inputFirst ? scratch.reservoir.second : scratch.reservoir.first;
            alpaka::onHost::memcpy(queue, boundaryTailSnapshot, vertexBatchScoreSum);
            scratch.clear(queue, output);
            if(domainPopulationCounts.empty())
                queue.enqueue(
                    relaunchFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AccumulateReflectedForwardPhiAse{},
                        hase::kernels::forward::TracePolicyList{
                            hase::kernels::forward::tracePolicy::source::surfaceReservoir,
                            hase::kernels::forward::tracePolicy::cell::forwardAse,
                            hase::kernels::forward::tracePolicy::boundary::surfaceReservoir,
                            positionPolicy,
                            diagnostics},
                        mesh,
                        populationRayCount,
                        rseBatch,
                        sourceWeight,
                        accumulation,
                        input.view(slotsPerFace),
                        scratch.samplingView(faceCount <= populationRayCount),
                        output.view(slotsPerFace),
                        rngSeed,
                        pass,
                        interfaceMap,
                        experiment.useReflections});
            else
                queue.enqueue(
                    relaunchFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AccumulateDomainCombedForwardPhiAseReservoir{},
                        hase::kernels::forward::TracePolicyList{
                            hase::kernels::forward::tracePolicy::source::surfaceReservoir,
                            hase::kernels::forward::tracePolicy::cell::forwardAse,
                            hase::kernels::forward::tracePolicy::boundary::surfaceReservoir,
                            positionPolicy,
                            diagnostics},
                        mesh,
                        populationRayCount,
                        rseBatch,
                        accumulation,
                        input.view(slotsPerFace),
                        scratch.domainComb.selectedView(populationRayCount),
                        scratch.domainComb.selectedWeightsView(populationRayCount),
                        output.view(slotsPerFace),
                        rngSeed,
                        pass,
                        interfaceMap,
                        experiment.useReflections});
            inputFirst = !inputFirst;

            double const currentWeight = updateSampling(output, pass);
            result.boundaryPasses = pass;
            result.boundaryRemainingFraction = currentWeight / initialWeight;
            residualWeightFractions.push_back(result.boundaryRemainingFraction);
            if(currentWeight > previousWeight)
            {
                ++growCount;
                if(growCount >= srmControls.divergenceStreak)
                {
                    auto const tail = estimateBoundaryTail(residualWeightFractions);
                    if(tail.divergent)
                    {
                        result.boundaryStatus = data::BoundaryStatus::diverged;
                        break;
                    }
                }
            }
            else
            {
                growCount = 0u;
                if(alpaka::math::abs(currentWeight - previousWeight) / alpaka::math::max(currentWeight, 1.0e-30)
                   < experiment.reflectionTolerance)
                {
                    result.boundaryStatus = data::BoundaryStatus::stable;
                    break;
                }
            }
            if(result.boundaryRemainingFraction < experiment.reflectionTolerance)
            {
                result.boundaryStatus = data::BoundaryStatus::converged;
                break;
            }
            previousWeight = currentWeight;
        }

        auto const tail = estimateBoundaryTail(residualWeightFractions);
        result.boundaryGamma = tail.gamma;
        result.boundaryGammaStandardError = tail.gammaStandardError;
        result.boundaryTailFactor = tail.tailFactor;
        result.boundaryTailClosure = tail.tailClosure;
        if(result.boundaryStatus == data::BoundaryStatus::diverged || tail.divergent)
        {
            result.boundaryStatus = data::BoundaryStatus::diverged;
        }
        else if(
            result.boundaryStatus == data::BoundaryStatus::stable
            || result.boundaryStatus == data::BoundaryStatus::maxPasses)
        {
            if(tail.applicable)
            {
                applyBoundaryTail(
                    queue,
                    devBundle.executor,
                    vertexBatchScoreSum,
                    boundaryTailSnapshot,
                    tail.tailFactor);
                result.boundaryStatus = data::BoundaryStatus::converged;
            }
        }
    }
} // namespace hase::core
