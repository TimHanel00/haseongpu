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
#include <core/geometry.hpp>
#include <kernels/forward/reflectionResampling.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hase::core
{
    /** @brief Device buffers for forward-ASE ray states prepared before traversal. */
    template<alpaka::onHost::concepts::Device T_Device>
    class PreparedRayTraceBank
    {
        using T_DoubleBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::uint32_t{1u}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device&>(), std::uint32_t{1u}));

    public:
        /**
         * @param device Device receiving all prepared-ray allocations.
         * @param maxRayCount Largest number of prepared ray states.
         */
        PreparedRayTraceBank(T_Device& device, std::uint32_t const maxRayCount)
            : components(
                  alpaka::onHost::alloc<double>(
                      device,
                      hase::kernels::forward::preparedRayTraceComponentCount * maxRayCount))
            , topology(
                  alpaka::onHost::alloc<std::uint32_t>(
                      device,
                      hase::kernels::forward::preparedRayTraceTopologyCount * maxRayCount))
            , maxRayCount(maxRayCount)
        {
        }

        /** @return Trivially-copyable non-owning view for device kernels. */
        [[nodiscard]] auto view()
        {
            auto result
                = hase::kernels::forward::PreparedRayTraceSpans{components.getView(), topology.getView(), maxRayCount};
            static_assert(std::is_trivially_copyable_v<ALPAKA_TYPEOF(result)>);
            static_assert(alpaka::concepts::KernelArg<ALPAKA_TYPEOF(result)>);
            return result;
        }

        T_DoubleBuffer components;
        T_UnsignedBuffer topology;
        std::uint32_t maxRayCount;
    };

    /** @brief Device buffers for one ray-indexed reflected-candidate bank. */
    template<alpaka::onHost::concepts::Device T_Device>
    class ReflectionCandidateBank
    {
        using T_DoubleBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::uint32_t{1u}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device&>(), std::uint32_t{1u}));

    public:
        /**
         * @param device Device receiving all candidate allocations.
         * @param maxRayCount Largest number of ray-owned candidates.
         */
        ReflectionCandidateBank(T_Device& device, std::uint32_t const maxRayCount)
            : components(
                  alpaka::onHost::alloc<double>(
                      device,
                      hase::kernels::forward::reflectionCandidateComponentCount * maxRayCount))
            , weights(alpaka::onHost::alloc<double>(device, maxRayCount))
            , faceIds(alpaka::onHost::alloc<std::uint32_t>(device, maxRayCount))
            , maxRayCount(maxRayCount)
        {
        }

        /** @return Trivially-copyable non-owning view for device kernels. */
        [[nodiscard]] auto view()
        {
            auto result = hase::kernels::forward::ReflectionCandidateSpans{
                components.getView(),
                weights.getView(),
                faceIds.getView(),
                maxRayCount};
            static_assert(std::is_trivially_copyable_v<ALPAKA_TYPEOF(result)>);
            static_assert(alpaka::concepts::KernelArg<ALPAKA_TYPEOF(result)>);
            return result;
        }

        T_DoubleBuffer components;
        T_DoubleBuffer weights;
        T_UnsignedBuffer faceIds;
        std::uint32_t maxRayCount;
    };

    /** @brief Persistent exact candidate and ordered-resampling storage for reflected ASE. */
    template<alpaka::onHost::concepts::Device T_Device>
    class ReflectionResamplingScratch
    {
        using T_DoubleBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::uint32_t{1u}));
        using T_ByteBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<char>(std::declval<T_Device&>(), std::uint32_t{1u}));
        using T_DoubleView = ALPAKA_TYPEOF(std::declval<T_DoubleBuffer&>().getView());
        using SamplingView = hase::kernels::forward::ReflectionSamplingSpans<T_DoubleView, T_DoubleView>;
        static_assert(std::is_trivially_copyable_v<SamplingView>);
        static_assert(alpaka::concepts::KernelArg<SamplingView>);

    public:
        /**
         * @param device Device receiving candidate and sampling allocations.
         * @param maxRayCount Largest supported launch ray count.
         */
        ReflectionResamplingScratch(T_Device device, std::uint32_t const maxRayCount)
            : preparedRays(device, maxRayCount)
            , first(device, maxRayCount)
            , second(device, maxRayCount)
            , samplingCdf(alpaka::onHost::alloc<double>(device, maxRayCount))
            , samplingTotalWeight(alpaka::onHost::alloc<double>(device, 1u))
            , samplingCdfScanBuffer(
                  alpaka::onHost::alloc<char>(
                      device,
                      alpaka::onHost::getScanBufferSize<double>(alpaka::Vec{maxRayCount})))
            , maxRayCount(maxRayCount)
        {
        }

        /**
         * @param rayCount Ray count required by the pending launch.
         * @throws std::runtime_error If the workspace is too small.
         */
        void validate(std::uint32_t const rayCount) const
        {
            if(rayCount > maxRayCount)
                throw std::runtime_error("reflection resampling scratch does not match this launch");
        }

        /** @brief Clear all ray-owned candidate weights in one bank. */
        void clear(concepts::Queue auto const& queue, ReflectionCandidateBank<T_Device>& bank)
        {
            alpaka::onHost::fill(queue, bank.weights, 0.0, bank.weights.getExtents());
        }

        /** @return Non-owning ordered CDF and total-weight views for kernels. */
        [[nodiscard]] SamplingView samplingView()
        {
            return {samplingCdf.getView(), samplingTotalWeight.getView()};
        }

        /**
         * @brief Build a cumulative weight array in ray-index order with a parallel scan.
         * @param devBundle Device and executor used for the transform and scan.
         * @param queue Queue targeting the workspace device.
         * @param bank Candidate bank supplying one weight per input ray.
         * @param rayCount Number of candidate slots participating in resampling.
         * @return Sum of candidate weights after queued work completes.
         */
        template<alpaka::concepts::Executor T_Executor>
        [[nodiscard]] double updateSampling(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            ReflectionCandidateBank<T_Device>& bank,
            std::uint32_t const rayCount)
        {
            auto candidateWeights = bank.weights.getView().getSubView(alpaka::Vec{rayCount});
            auto cdf = samplingCdf.getView().getSubView(alpaka::Vec{rayCount});
            alpaka::onHost::transform(
                queue,
                devBundle.executor,
                cdf,
                alpaka::ScalarFunc{hase::kernels::forward::FilterReflectionSamplingWeight{}},
                candidateWeights);
            alpaka::onHost::inclusiveScanInPlace(queue, devBundle.executor, samplingCdfScanBuffer, cdf);

            auto const scalarFrameSpec = hase::alpakaUtils::getFrameSpec<std::uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{1u});
            queue.enqueue(
                scalarFrameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::CaptureReflectionSamplingTotalWeight{},
                    rayCount,
                    samplingView()});

            std::array<double, 1u> totalWeightHost{};
            hase::concepts::HybridBuffer<double> auto totalWeight
                = hase::alpakaUtils::getHybridBuffer(totalWeightHost, samplingTotalWeight);
            totalWeight.toHost(queue);
            return totalWeight.getHostView()[0u];
        }

        PreparedRayTraceBank<T_Device> preparedRays;
        ReflectionCandidateBank<T_Device> first;
        ReflectionCandidateBank<T_Device> second;
        T_DoubleBuffer samplingCdf;
        T_DoubleBuffer samplingTotalWeight;
        T_ByteBuffer samplingCdfScanBuffer;
        std::uint32_t maxRayCount;
    };
} // namespace hase::core
