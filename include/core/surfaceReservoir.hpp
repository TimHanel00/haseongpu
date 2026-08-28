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
#include <core/boundaryRayBuffer.hpp>
#include <core/geometry.hpp>
#include <core/particleComb.hpp>
#include <data/AseDomainGraph.hpp>
#include <kernels/forward/surfaceReservoir.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hase::core
{
    template<
        alpaka::onHost::concepts::Device T_Device,
        hase::kernels::forward::SurfaceReservoirPositionPolicy T_PositionPolicy>
    class SurfaceReservoirPositionBuffers;

    template<alpaka::onHost::concepts::Device T_Device>
    class SurfaceReservoirPositionBuffers<T_Device, hase::kernels::forward::surfaceReservoirPosition::Exact>
    {
    public:
        SurfaceReservoirPositionBuffers(
            T_Device& device,
            std::uint32_t const reservoirSlotCount,
            std::uint32_t const maxCandidateCount)
            : positions(device, reservoirSlotCount)
            , candidatePositions(device, maxCandidateCount)
        {
        }

        [[nodiscard]] auto view()
        {
            return hase::kernels::forward::ExactSurfaceReservoirPositionSpans{
                positions.view(),
                candidatePositions.view()};
        }

    private:
        hase::core::PositionBufferSoA<T_Device> positions;
        hase::core::PositionBufferSoA<T_Device> candidatePositions;
    };

    template<alpaka::onHost::concepts::Device T_Device>
    class SurfaceReservoirPositionBuffers<T_Device, hase::kernels::forward::surfaceReservoirPosition::Centroid>
    {
    public:
        SurfaceReservoirPositionBuffers(T_Device&, std::uint32_t, std::uint32_t)
        {
        }

        [[nodiscard]] static constexpr hase::kernels::forward::CentroidSurfaceReservoirPositionSpans view()
        {
            return {};
        }
    };

    /** @brief Device buffers for one per-face weighted surface-reservoir bank. */
    template<
        alpaka::onHost::concepts::Device T_Device,
        hase::kernels::forward::SurfaceReservoirPositionPolicy T_PositionPolicy
        = hase::kernels::forward::surfaceReservoirPosition::Exact>
    class SurfaceReservoirBank
    {
        static_assert(
            !std::same_as<T_PositionPolicy, hase::kernels::forward::surfaceReservoirPosition::Centroid>
            || std::is_empty_v<SurfaceReservoirPositionBuffers<T_Device, T_PositionPolicy>>);

        using T_DoubleBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_KeyBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint64_t>(std::declval<T_Device&>(), std::size_t{1u}));

    public:
        /**
         * @param device Device receiving all reservoir allocations.
         * @param faceCount Number of local cell faces.
         * @param maxCandidateCount Largest number of ray-owned candidates.
         */
        SurfaceReservoirBank(
            T_Device& device,
            std::uint32_t faceCount,
            alpaka::concepts::Vector auto slotsPerFace,
            std::uint32_t maxCandidateCount)
            : counts(alpaka::onHost::alloc<std::uint32_t>(device, static_cast<std::size_t>(faceCount)))
            , positionBuffers(
                  device,
                  static_cast<std::uint32_t>(static_cast<std::size_t>(faceCount) * slotsPerFace.x()),
                  maxCandidateCount)
            , directions(device, static_cast<std::size_t>(faceCount) * slotsPerFace.x())
            , weights(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(faceCount) * slotsPerFace.x()))
            , wavelengths(
                  alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(faceCount) * slotsPerFace.x()))
            , faceWeights(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(faceCount)))
            , selectionKeys(
                  alpaka::onHost::alloc<std::uint64_t>(device, static_cast<std::size_t>(faceCount) * slotsPerFace.x()))
            , candidateDirections(device, static_cast<std::size_t>(maxCandidateCount))
            , candidateWeights(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(maxCandidateCount)))
            , candidateWavelengths(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(maxCandidateCount)))
        {
        }

        /**
         * @return Trivially-copyable non-owning view for device kernels.
         */
        [[nodiscard]] auto view(alpaka::concepts::Vector auto slotsPerFace)
        {
            auto result = hase::kernels::forward::SurfaceReservoirSpans{
                counts.getView(),
                positionBuffers.view(),
                directions.view(),
                weights.getView(),
                wavelengths.getView(),
                faceWeights.getView(),
                selectionKeys.getView(),
                candidateDirections.view(),
                candidateWeights.getView(),
                candidateWavelengths.getView(),
                slotsPerFace};
            static_assert(std::is_trivially_copyable_v<ALPAKA_TYPEOF(result)>);
            static_assert(alpaka::concepts::KernelArg<ALPAKA_TYPEOF(result)>);
            return result;
        }

        T_UnsignedBuffer counts;
        SurfaceReservoirPositionBuffers<T_Device, T_PositionPolicy> positionBuffers;
        hase::core::DirectionBufferSoA<T_Device> directions;
        T_DoubleBuffer weights;
        T_DoubleBuffer wavelengths;
        T_DoubleBuffer faceWeights;
        T_KeyBuffer selectionKeys;
        hase::core::DirectionBufferSoA<T_Device> candidateDirections;
        T_DoubleBuffer candidateWeights;
        T_DoubleBuffer candidateWavelengths;
    };

    /** @brief Pair of reservoir banks used as alternating SRM input and output. */
    template<
        alpaka::onHost::concepts::Device T_Device,
        hase::kernels::forward::SurfaceReservoirPositionPolicy T_PositionPolicy
        = hase::kernels::forward::surfaceReservoirPosition::Exact>
    class SurfaceReservoir
    {
    public:
        /**
         * @param device Device receiving both banks.
         * @param faceCount Number of local cell faces.
         * @param maxCandidateCount Largest number of ray-owned candidates per bank.
         */
        SurfaceReservoir(
            T_Device& device,
            std::uint32_t faceCount,
            alpaka::concepts::Vector auto slotsPerFace,
            std::uint32_t maxCandidateCount)
            : first(device, faceCount, slotsPerFace, maxCandidateCount)
            , second(device, faceCount, slotsPerFace, maxCandidateCount)
            , faceCount(faceCount)
        {
        }

        SurfaceReservoirBank<T_Device, T_PositionPolicy> first;
        SurfaceReservoirBank<T_Device, T_PositionPolicy> second;
        std::uint32_t faceCount;
    };

    /** @brief Persistent reservoir and scan workspace for one maximum launch shape. */
    template<
        alpaka::onHost::concepts::Device T_Device,
        hase::kernels::forward::SurfaceReservoirPositionPolicy T_PositionPolicy
        = hase::kernels::forward::surfaceReservoirPosition::Exact>
    class SurfaceReservoirScratch
    {
        using T_DoubleBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_ByteBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<char>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_TotalWeightTransfer = alpakaUtils::GetHybridBuffer_t<T_Device, std::array<double, 1u>>;
        using T_DoubleView = ALPAKA_TYPEOF(std::declval<T_DoubleBuffer&>().getView());
        using T_UnsignedView = ALPAKA_TYPEOF(std::declval<T_UnsignedBuffer&>().getView());
        using SamplingView
            = hase::kernels::forward::SurfaceReservoirSamplingCdfSpans<T_DoubleView, T_DoubleView, T_UnsignedView>;
        static_assert(std::is_trivially_copyable_v<SamplingView>);
        static_assert(alpaka::concepts::KernelArg<SamplingView>);

    public:
        /**
         * @param device Device receiving reservoir and scan allocations.
         * @param faceCount Number of local cell faces.
         * @param maxRayCount Largest supported launch ray count.
         */
        SurfaceReservoirScratch(
            T_Device device,
            std::uint32_t faceCount,
            alpaka::concepts::Vector auto slotsPerFace,
            std::uint32_t maxRayCount)
            : reservoir(device, faceCount, slotsPerFace, boundaryCandidateCount(maxRayCount))
            , domainComb(device, faceCount, maxRayCount)
            , samplingCdf(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(faceCount)))
            , samplingTotalWeight(alpaka::onHost::alloc<double>(device, std::size_t{1u}))
            , systematicOffset(alpaka::onHost::alloc<double>(device, std::size_t{1u}))
            , stratifiedRayCounts(alpaka::onHost::alloc<std::uint32_t>(device, static_cast<std::size_t>(faceCount)))
            , stratifiedRayOffsets(alpaka::onHost::alloc<std::uint32_t>(device, static_cast<std::size_t>(faceCount)))
            , stratifiedRayFaces(alpaka::onHost::alloc<std::uint32_t>(device, static_cast<std::size_t>(maxRayCount)))
            , samplingCdfScanBuffer(
                  alpaka::onHost::alloc<char>(
                      device,
                      alpaka::onHost::getScanBufferSize<double>(alpaka::Vec{static_cast<std::size_t>(faceCount)})))
            , stratifiedCountScanBuffer(
                  alpaka::onHost::alloc<char>(
                      device,
                      alpaka::onHost::getScanBufferSize<std::uint32_t>(
                          alpaka::Vec{static_cast<std::size_t>(faceCount)})))
            , stratifiedFaceScanBuffer(
                  alpaka::onHost::alloc<char>(
                      device,
                      alpaka::onHost::getScanBufferSize<std::uint32_t>(
                          alpaka::Vec{static_cast<std::size_t>(maxRayCount)})))
            , totalWeightHost{}
            , totalWeightTransfer(alpakaUtils::getHybridBuffer(device, totalWeightHost))
            , maxRayCount(maxRayCount)
        {
        }

        /**
         * @param faceCount Face count required by the pending launch.
         * @param rayCount Ray count required by the pending launch.
         * @throws std::runtime_error If the workspace shape is incompatible.
         */
        void validate(std::uint32_t const faceCount, std::uint32_t const rayCount) const
        {
            if(reservoir.faceCount != faceCount || rayCount > maxRayCount)
                throw std::runtime_error("surface reservoir scratch does not match this launch");
        }

        /**
         * @brief Enqueue clearing of counts and aggregate face weights for one bank.
         * @param queue Queue targeting the reservoir device.
         * @param bank Bank whose per-face accumulation state is reset.
         */
        void clear(concepts::Queue auto const& queue, SurfaceReservoirBank<T_Device, T_PositionPolicy>& bank)
        {
            auto const extent = alpaka::Vec{static_cast<std::size_t>(reservoir.faceCount)};
            alpaka::onHost::fill(queue, bank.counts, 0u, extent);
            alpaka::onHost::fill(queue, bank.faceWeights, 0.0, extent);
            alpaka::onHost::fill(
                queue,
                bank.selectionKeys,
                std::numeric_limits<std::uint64_t>::max(),
                bank.selectionKeys.getExtents());
        }

        /**
         * @param useFaceStratification Whether ray-to-face assignments are precomputed.
         * @return Non-owning CDF and optional stratified-face views for kernels.
         */
        [[nodiscard]] SamplingView samplingView(bool const useFaceStratification)
        {
            return {
                samplingCdf.getView(),
                samplingTotalWeight.getView(),
                stratifiedRayFaces.getView(),
                useFaceStratification};
        }

        /**
         * @brief Build the normalized face CDF and optional systematic assignments.
         * @param devBundle Device and executor used for scans and kernels.
         * @param queue Queue targeting the workspace device.
         * @param bank Reservoir bank supplying aggregate face weights.
         * @param rayCount Number of rays that will sample the bank.
         * @param rngSeed Run-level seed for systematic stratification.
         * @param pass One-based SRM pass index.
         * @return Sum of the bank's aggregate face weights after queued work completes.
         */
        template<alpaka::concepts::Executor T_Executor>
        [[nodiscard]] double updateSampling(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            SurfaceReservoirBank<T_Device, T_PositionPolicy>& bank,
            alpaka::concepts::Vector auto slotsPerFace,
            std::uint32_t const rayCount,
            std::uint32_t const rngSeed,
            std::uint32_t const pass)
        {
            bool const useFaceStratification = reservoir.faceCount <= rayCount;
            auto const reservoirView = bank.view(slotsPerFace);
            SamplingView const cdfView = samplingView(useFaceStratification);
            auto const faceFrameSpec = hase::alpakaUtils::getFrameSpec<std::uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{reservoir.faceCount});
            auto const scalarFrameSpec = hase::alpakaUtils::getFrameSpec<std::uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{1u});

            queue.enqueue(
                faceFrameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::FinalizeSurfaceReservoir{},
                    reservoir.faceCount,
                    reservoirView});

            alpaka::onHost::inclusiveScan(
                queue,
                devBundle.executor,
                samplingCdfScanBuffer,
                samplingCdf,
                reservoirView.faceWeights);
            queue.enqueue(
                scalarFrameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::CaptureSurfaceReservoirSamplingTotalWeight{},
                    reservoir.faceCount,
                    cdfView});
            queue.enqueue(
                faceFrameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::NormalizeSurfaceReservoirSamplingCdf{},
                    reservoir.faceCount,
                    cdfView});
            if(useFaceStratification)
            {
                queue.enqueue(
                    scalarFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::GenerateSurfaceReservoirSystematicOffset{},
                        systematicOffset,
                        rngSeed,
                        pass});
                queue.enqueue(
                    faceFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AssignSurfaceReservoirStratifiedRayCounts{},
                        reservoir.faceCount,
                        rayCount,
                        cdfView,
                        systematicOffset,
                        stratifiedRayCounts});
                alpaka::onHost::exclusiveScan(
                    queue,
                    devBundle.executor,
                    stratifiedCountScanBuffer,
                    stratifiedRayOffsets,
                    stratifiedRayCounts);
                auto const rayExtent = alpaka::Vec{static_cast<std::size_t>(rayCount)};
                auto rayFaces = stratifiedRayFaces.getView().getSubView(rayExtent);
                alpaka::onHost::fill(queue, rayFaces, 0u, rayExtent);
                queue.enqueue(
                    faceFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::MarkSurfaceReservoirStratifiedFaceStarts{},
                        reservoir.faceCount,
                        rayCount,
                        stratifiedRayOffsets,
                        rayFaces});
                alpaka::onHost::inclusiveScanInPlace(queue, devBundle.executor, stratifiedFaceScanBuffer, rayFaces);
            }

            alpaka::onHost::memcpy(queue, totalWeightTransfer.toDeviceView(), samplingTotalWeight);
            totalWeightTransfer.toHost(queue);
            return totalWeightTransfer.getHostView()[0u];
        }

        /** @brief Finalize one bank and enqueue fixed per-domain face populations. */
        template<alpaka::concepts::Executor T_Executor>
        [[nodiscard]] double updateDomainSampling(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            SurfaceReservoirBank<T_Device, T_PositionPolicy>& bank,
            alpaka::concepts::Vector auto slotsPerFace,
            hase::data::AseDomainInterfaceView const interfaceMap,
            std::span<std::uint32_t const> const domainPopulationCounts,
            std::uint32_t const facesPerCell,
            std::uint32_t const rngSeed,
            std::uint32_t const pass)
        {
            auto const faceFrameSpec = hase::alpakaUtils::getFrameSpec<std::uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{reservoir.faceCount});
            queue.enqueue(
                faceFrameSpec,
                alpaka::KernelBundle{
                    hase::kernels::forward::FinalizeSurfaceReservoir{},
                    reservoir.faceCount,
                    bank.view(slotsPerFace)});
            std::uint32_t outputOffset = 0u;
            for(std::uint32_t domain = 0u; domain < domainPopulationCounts.size(); ++domain)
            {
                auto const count = domainPopulationCounts[domain];
                if(count == 0u)
                    continue;
                domainComb.enqueueFaceDomain(
                    devBundle,
                    queue,
                    bank.faceWeights.getView(),
                    interfaceMap.cellDomains,
                    reservoir.faceCount,
                    facesPerCell,
                    domain,
                    outputOffset,
                    count,
                    rngSeed,
                    (static_cast<std::uint64_t>(pass) << 32u) | domain);
                outputOffset += count;
            }
            if(outputOffset > maxRayCount)
                throw std::runtime_error("SRM domain populations exceed the relaunch capacity");
            domainComb.enqueueTotal(devBundle, queue, bank.faceWeights.getView(), reservoir.faceCount);
            alpaka::onHost::memcpy(queue, totalWeightTransfer.toDeviceView(), domainComb.totalWeight);
            totalWeightTransfer.toHost(queue);
            return totalWeightTransfer.getHostView()[0u];
        }

        SurfaceReservoir<T_Device, T_PositionPolicy> reservoir;
        ParticleCombWorkspace<T_Device> domainComb;
        T_DoubleBuffer samplingCdf;
        T_DoubleBuffer samplingTotalWeight;
        T_DoubleBuffer systematicOffset;
        T_UnsignedBuffer stratifiedRayCounts;
        T_UnsignedBuffer stratifiedRayOffsets;
        T_UnsignedBuffer stratifiedRayFaces;
        T_ByteBuffer samplingCdfScanBuffer;
        T_ByteBuffer stratifiedCountScanBuffer;
        T_ByteBuffer stratifiedFaceScanBuffer;
        std::array<double, 1u> totalWeightHost;
        T_TotalWeightTransfer totalWeightTransfer;
        std::uint32_t maxRayCount;
    };
} // namespace hase::core
