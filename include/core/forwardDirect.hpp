#pragma once

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/HybridBuffer.hpp>
#include <concepts/concepts.hpp>
#include <core/Runtime.hpp>
#include <core/boundaryRayBuffer.hpp>
#include <core/forwardSrm.hpp>
#include <core/particleComb.hpp>
#include <kernels/forward/directBoundary.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace hase::core
{
    /** @brief Persistent device buffers for population-controlled direct boundary tracing. */
    template<alpaka::onHost::concepts::Device T_Device>
    struct DirectBoundaryScratch
    {
        using T_TotalWeightBuffer = alpakaUtils::GetHybridBuffer_t<T_Device, std::array<double, 1u>>;

        DirectBoundaryScratch(T_Device& device, std::uint32_t const capacity)
            : first(device, boundaryCandidateCount(capacity))
            , second(device, boundaryCandidateCount(capacity))
            , comb(device, boundaryCandidateCount(capacity), capacity)
            , totalWeightHost{}
            , totalWeight(alpakaUtils::getHybridBuffer(device, totalWeightHost))
            , m_capacity(capacity)
        {
        }

        void validate(std::uint32_t const rayCount) const
        {
            if(rayCount > m_capacity)
                throw std::out_of_range("direct boundary scratch capacity exceeded");
        }

        BoundaryRayBuffer<T_Device> first;
        BoundaryRayBuffer<T_Device> second;
        ParticleCombWorkspace<T_Device> comb;
        std::array<double, 1u> totalWeightHost;
        T_TotalWeightBuffer totalWeight;

    private:
        std::uint32_t m_capacity;
    };

    /**
     * @brief Trace direct boundary passes while combing every outgoing population on the device.
     *
     * Large candidate, prefix, selection, and relaunch buffers remain device-resident. One scalar
     * total is downloaded at pass boundaries for convergence and pass termination.
     */
    template<alpaka::onHost::concepts::Device T_Device, alpaka::concepts::Executor T_Exec>
    void runForwardDirect(
        alpakaUtils::DevBundle<T_Device, T_Exec>& devBundle,
        concepts::Queue auto const& queue,
        hase::data::TraceView const mesh,
        AseTraceControls const& experiment,
        ForwardPhiAseRawResult& result,
        unsigned const rayCount,
        unsigned const rseBatch,
        double const sourceStrengthTotal,
        alpaka::concepts::IBuffer<double> auto& vertexBatchScoreSum,
        alpaka::concepts::IBuffer<std::uint32_t> auto& volumeRayVisits,
        alpaka::concepts::IBuffer<std::uint32_t> auto& droppedRays,
        unsigned const rngSeed,
        DirectBoundaryScratch<T_Device>& scratch,
        hase::kernels::forward::concepts::TracePolicy auto const diagnostics,
        hase::data::AseDomainInterfaceView const interfaceMap = {},
        hase::data::AseDomainSourceView const domainSources = {},
        std::span<std::uint32_t const> const domainRayCounts = {},
        std::span<double const> const domainSourceWeights = {},
        std::span<std::uint32_t const> const domainPopulationCounts = {})
    {
        scratch.validate(rayCount);
        std::uint32_t populationRayCount = rayCount;
        if(!domainPopulationCounts.empty())
        {
            populationRayCount = 0u;
            for(auto const count : domainPopulationCounts)
                populationRayCount += count;
            if(populationRayCount == 0u && rayCount != 0u)
                throw std::invalid_argument("a non-empty direct launch requires a relaunch population");
            scratch.validate(populationRayCount);
        }
        auto accumulation = hase::kernels::forward::ForwardAccumulationSpans{
            vertexBatchScoreSum.getMdSpan(),
            volumeRayVisits.getMdSpan(),
            droppedRays.getMdSpan()};
        auto const frameSpec = getRayFrameSpec(rayCount, queue);
        auto readTotalWeight = [&]()
        {
            alpaka::onHost::memcpy(queue, scratch.totalWeight.toDeviceView(), scratch.comb.totalWeight);
            scratch.totalWeight.toHost(queue);
            return scratch.totalWeight.getHostView()[0u];
        };

        auto enqueueComb = [&](auto& candidates, std::uint32_t const candidateCount, std::uint32_t const pass)
        {
            if(domainPopulationCounts.empty())
            {
                scratch.comb.enqueue(
                    devBundle,
                    queue,
                    candidates.weights.getView(),
                    candidateCount,
                    populationRayCount,
                    rngSeed,
                    static_cast<std::uint64_t>(pass));
                return;
            }
            std::uint32_t outputOffset = 0u;
            for(std::uint32_t domain = 0u; domain < domainPopulationCounts.size(); ++domain)
            {
                auto const count = domainPopulationCounts[domain];
                if(count == 0u)
                    continue;
                scratch.comb.enqueueDomain(
                    devBundle,
                    queue,
                    candidates.weights.getView(),
                    candidates.targetDomains.getView(),
                    candidateCount,
                    domain,
                    outputOffset,
                    count,
                    rngSeed,
                    (static_cast<std::uint64_t>(pass) << 32u) | domain);
                outputOffset += count;
            }
            if(outputOffset != populationRayCount)
                throw std::runtime_error("domain population counts do not match the relaunch population");
            scratch.comb.enqueueTotal(devBundle, queue, candidates.weights.getView(), candidateCount);
        };

        auto const initialCandidateCount = boundaryCandidateCount(rayCount);
        alpaka::onHost::fill(
            queue,
            scratch.first.weights,
            0.0,
            alpaka::Vec{static_cast<std::size_t>(initialCandidateCount)});
        if(domainRayCounts.empty())
            queue.enqueue(
                frameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::AccumulateForwardPhiAseDirect{},
                    hase::kernels::forward::TracePolicyList{
                        hase::kernels::forward::tracePolicy::source::volume,
                        hase::kernels::forward::tracePolicy::cell::forwardAse,
                        hase::kernels::forward::tracePolicy::boundary::boundaryCandidates,
                        hase::kernels::forward::tracePolicy::position::exact,
                        diagnostics},
                    mesh,
                    rayCount,
                    rseBatch,
                    sourceStrengthTotal,
                    accumulation,
                    scratch.first.view(),
                    rngSeed,
                    interfaceMap,
                    experiment.useReflections});
        else
        {
            unsigned candidateOffset = 0u;
            for(std::uint32_t domain = 0u; domain < domainRayCounts.size(); ++domain)
            {
                auto const count = domainRayCounts[domain];
                if(count == 0u)
                    continue;
                auto const domainFrame = getRayFrameSpec(count, queue);
                double const sourceWeight = domainSourceWeights[domain];
                queue.enqueue(
                    domainFrame,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AccumulateDomainForwardPhiAseDirect{},
                        hase::kernels::forward::TracePolicyList{
                            hase::kernels::forward::tracePolicy::source::volume,
                            hase::kernels::forward::tracePolicy::cell::forwardAse,
                            hase::kernels::forward::tracePolicy::boundary::boundaryCandidates,
                            hase::kernels::forward::tracePolicy::position::exact,
                            diagnostics},
                        mesh,
                        domainSources,
                        domain,
                        count,
                        candidateOffset,
                        rseBatch,
                        sourceWeight,
                        accumulation,
                        scratch.first.view(),
                        rngSeed,
                        interfaceMap,
                        experiment.useReflections});
                candidateOffset += count;
            }
            if(candidateOffset != rayCount)
                throw std::runtime_error("domain batch ray counts do not match the launch total");
        }
        enqueueComb(scratch.first, initialCandidateCount, 0u);

        double const initialWeight = readTotalWeight();
        result.boundaryMaxPasses = experiment.resolvedBoundaryMaxPasses(experiment.domainCount);
        if(initialWeight <= 0.0)
        {
            result.boundaryStatus = data::BoundaryStatus::converged;
            return;
        }

        bool inputFirst = true;
        double previousWeight = initialWeight;
        unsigned growCount = 0u;
        constexpr unsigned divergenceStreak = 3u;
        result.boundaryDivergenceStreak = divergenceStreak;
        result.boundaryStatus = data::BoundaryStatus::maxPasses;
        result.boundaryRemainingFraction = 1.0;
        auto const relaunchFrame = getRayFrameSpec(populationRayCount, queue);
        auto const relaunchCandidateCount = boundaryCandidateCount(populationRayCount);
        for(unsigned pass = 1u; pass <= result.boundaryMaxPasses; ++pass)
        {
            auto& output = inputFirst ? scratch.second : scratch.first;
            auto& input = inputFirst ? scratch.first : scratch.second;
            alpaka::onHost::fill(
                queue,
                output.weights,
                0.0,
                alpaka::Vec{static_cast<std::size_t>(relaunchCandidateCount)});
            queue.enqueue(
                relaunchFrame,
                alpaka::KernelBundle{
                    hase::kernels::forward::AccumulateParticleCombedForwardPhiAse{},
                    hase::kernels::forward::TracePolicyList{
                        hase::kernels::forward::tracePolicy::source::boundaryCandidates,
                        hase::kernels::forward::tracePolicy::cell::forwardAse,
                        hase::kernels::forward::tracePolicy::boundary::boundaryCandidates,
                        hase::kernels::forward::tracePolicy::position::exact,
                        diagnostics},
                    mesh,
                    populationRayCount,
                    accumulation,
                    input.view(),
                    scratch.comb.selectedView(populationRayCount),
                    scratch.comb.selectedWeightsView(populationRayCount),
                    output.view(),
                    interfaceMap,
                    pass,
                    experiment.useReflections});
            enqueueComb(output, relaunchCandidateCount, pass);
            double const currentWeight = readTotalWeight();
            result.boundaryPasses = pass;
            result.boundaryRemainingFraction = currentWeight / initialWeight;
            if(currentWeight <= 0.0 || result.boundaryRemainingFraction < experiment.reflectionTolerance)
            {
                result.boundaryStatus = data::BoundaryStatus::converged;
                break;
            }
            if(currentWeight > previousWeight)
            {
                ++growCount;
                if(growCount >= divergenceStreak)
                {
                    result.boundaryStatus = data::BoundaryStatus::diverged;
                    break;
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
            previousWeight = currentWeight;
            inputFirst = !inputFirst;
        }
    }
} // namespace hase::core
