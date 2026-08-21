/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpakaUtils/HybridBuffer.hpp>
#include <alpakaUtils/memory.hpp>
#include <benchmark.hpp>
#include <core/forwardSrm.hpp>
#include <data/TraceData.hpp>
#include <kernels/forwardPhiAseMapping.hpp>
#include <kernels/vertexAccumulation.hpp>
#include <random/random.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include <vector>

namespace hase::core
{
    struct BetaVolumeContribution
    {
        constexpr auto operator()(alpaka::concepts::Simd auto const& beta, alpaka::concepts::Simd auto const& volume)
            const
        {
            return beta * alpaka::pCast<double>(volume);
        }
    };

    [[nodiscard]] ForwardPhiAseRawResult makeForwardRawResult(
        unsigned volumeCount,
        unsigned vertexCount,
        unsigned batchCount = kernels::forward::defaultForwardRseBatchCount);

    [[nodiscard]] double calcForwardSourceStrengthTotal(hase::data::TraceData const& trace);

    void mergeForwardRawResult(ForwardPhiAseRawResult& target, ForwardPhiAseRawResult const& source);

    [[nodiscard]] double calcForwardRelativeStandardError(double scoreSum, double scoreSquareSum, unsigned rayCount);

    [[nodiscard]] double calcForwardStandardError(
        double scoreSum,
        double scoreSquareSum,
        unsigned rayCount,
        double normalizationVolume,
        double volumeSize);

    void finalizeForwardPhiAse(
        hase::data::TraceData const& hostMesh,
        ForwardPhiAseRawResult const& rawResult,
        data::PhiAseResult& result);

    void finalizeForwardPhiAse(
        hase::data::TraceData const& hostMesh,
        ForwardPhiAseRawResult const& rawResult,
        double betaVolumeTotal,
        data::PhiAseResult& result);

    template<alpaka::onHost::concepts::Device T_Device, typename T_Exec>
    class ForwardPhiAseDeviceContext
    {
        using T_Queue = ALPAKA_TYPEOF(std::declval<T_Device>().makeQueue(alpaka::queueKind::nonBlocking));
        using T_DoubleBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1}));
        using T_FloatBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<float>(std::declval<T_Device&>(), std::size_t{1}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<unsigned>(std::declval<T_Device&>(), std::size_t{1}));
        using T_CharBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<char>(std::declval<T_Device&>(), std::size_t{1}));
        using T_RseBatchRayCounts = hase::alpakaUtils::GetHybridBuffer_t<T_Device, std::vector<unsigned>>;
        using T_BetaVolumeTotal = hase::alpakaUtils::GetHybridBuffer_t<T_Device, std::array<double, 1u>>;

    public:
        ForwardPhiAseDeviceContext(
            T_Device const& device,
            T_Exec const& executor,
            AseTraceControls const& experiment,
            hase::data::TraceData const& hostMesh)
            : m_devBundle(device, executor)
            , m_queue(m_devBundle.device.makeQueue(alpaka::queueKind::nonBlocking))
            , m_rseBatchRayCounts(hase::kernels::forward::defaultForwardRseBatchCount, 0u)
            , m_rseBatchRayCountsBuffer(hase::alpakaUtils::getHybridBuffer(m_devBundle.device, m_rseBatchRayCounts))
            , m_sourceStrengthTotalHost{}
            , m_sourceStrengthTotal(hase::alpakaUtils::getHybridBuffer(m_devBundle.device, m_sourceStrengthTotalHost))
            , m_vertexBatchScoreSum(
                  alpaka::onHost::alloc<double>(
                      m_devBundle.device,
                      hase::kernels::forward::defaultForwardRseBatchCount * hostMesh.numberOfMaterials
                          * static_cast<std::size_t>(hostMesh.numberOfMeshPoints)))
            , m_volumeRayVisits(
                  alpaka::onHost::alloc<unsigned>(
                      m_devBundle.device,
                      static_cast<std::size_t>(hostMesh.numberOfCells)))
            , m_droppedRays(
                  alpaka::onHost::alloc<unsigned>(
                      m_devBundle.device,
                      static_cast<std::size_t>(hostMesh.numberOfCells)))
            , m_lumpedMaterialVertexVolume(
                  hase::alpakaUtils::toDevice(m_queue, hase::kernels::makeLumpedMaterialVertexVolumes(hostMesh)))
            , m_volumePhiAse(
                  alpaka::onHost::alloc<float>(m_devBundle.device, static_cast<std::size_t>(hostMesh.numberOfCells)))
            , m_standardError(
                  alpaka::onHost::alloc<double>(m_devBundle.device, static_cast<std::size_t>(hostMesh.numberOfCells)))
            , m_relativeStandardError(
                  alpaka::onHost::alloc<double>(m_devBundle.device, static_cast<std::size_t>(hostMesh.numberOfCells)))
            , m_volumeDndtAse(
                  alpaka::onHost::alloc<double>(m_devBundle.device, static_cast<std::size_t>(hostMesh.numberOfCells)))
            , m_sourceStrengthWeights(
                  alpaka::onHost::alloc<double>(m_devBundle.device, static_cast<std::size_t>(hostMesh.numberOfCells)))
            , m_sourceStrengthPrefixScanBuffer(
                  alpaka::onHost::alloc<char>(
                      m_devBundle.device,
                      alpaka::onHost::getScanBufferSize<double>(
                          alpaka::Vec{static_cast<std::size_t>(hostMesh.numberOfCells)})))
            , m_volumeCount(hostMesh.numberOfCells)
            , m_materialVertexCount(hostMesh.numberOfMaterials * hostMesh.numberOfMeshPoints)
            , m_batchCount(hase::kernels::forward::defaultForwardRseBatchCount)
        {
            if(experiment.useReflections)
            {
                m_srmWorkspace = std::make_unique<ForwardSrmWorkspace<T_Device>>(
                    m_devBundle.device,
                    m_volumeCount * hase::data::tet4FaceCount,
                    experiment.surfaceReservoirSize,
                    std::max(experiment.maxRays, experiment.resolvedForwardRayCount()));
            }
        }

        /** @brief Resize persistent batch accumulation storage for a worker group. */
        void configureBatchCount(unsigned const batchCount)
        {
            if(batchCount == 0u)
                throw std::invalid_argument("forward ASE batch count must be positive");
            if(batchCount == m_batchCount)
                return;
            alpaka::onHost::wait(m_queue);
            m_vertexBatchScoreSum = alpaka::onHost::alloc<double>(
                m_devBundle.device,
                batchCount * static_cast<std::size_t>(m_materialVertexCount));
            m_batchCount = batchCount;
            m_rseBatchRayCounts.assign(batchCount, 0u);
            m_rseBatchRayCountsBuffer = hase::alpakaUtils::getHybridBuffer(m_devBundle.device, m_rseBatchRayCounts);
        }

        void begin(
            hase::data::TraceView const mesh,
            unsigned rayCount,
            unsigned rngSeed,
            unsigned rseBatch,
            double betaVolumeTotal,
            AseTraceControls const& experiment,
            bool resetAccumulators = true)
        {
            m_started = std::chrono::steady_clock::now();
            m_rayCount = rayCount;
            if(resetAccumulators)
            {
                m_accumulatedRayCount = 0u;
                std::ranges::fill(m_rseBatchRayCounts, 0u);
            }
            m_accumulatedRayCount += rayCount;
            if(rseBatch >= m_batchCount)
                throw std::out_of_range("forward ASE RSE batch index is out of range");
            m_rseBatchRayCounts.at(rseBatch) += rayCount;
            if(rayCount == 0u)
                return;

            if(resetAccumulators)
            {
                alpaka::onHost::fill(
                    m_queue,
                    m_vertexBatchScoreSum,
                    0.0,
                    alpaka::Vec{m_batchCount * static_cast<std::size_t>(m_materialVertexCount)});
                alpaka::onHost::fill(
                    m_queue,
                    m_volumeRayVisits,
                    0u,
                    alpaka::Vec{static_cast<std::size_t>(m_volumeCount)});
                alpaka::onHost::fill(m_queue, m_droppedRays, 0u, alpaka::Vec{static_cast<std::size_t>(m_volumeCount)});
            }

            auto accumulation = kernels::forward::ForwardAccumulationSpans{
                m_vertexBatchScoreSum.getMdSpan(),
                m_volumeRayVisits.getMdSpan(),
                m_droppedRays.getMdSpan()};
            auto frameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
                m_devBundle.device,
                m_devBundle.executor,
                alpaka::Vec{rayCount});
            m_srmResult = makeForwardRawResult(m_volumeCount, m_materialVertexCount, m_batchCount);
            m_srmResult.rayCount = rayCount;
            if(experiment.useReflections)
            {
                if(!m_srmWorkspace)
                    throw std::runtime_error("persistent forward SRM workspace was not initialized");
                auto const controls = resolveSrmControls(experiment);
                m_srmResult.srmMaxIterations = controls.maxIterations;
                m_srmResult.srmDivergenceStreak = controls.divergenceStreak;
                runForwardSrm(
                    m_devBundle,
                    m_queue,
                    mesh,
                    experiment,
                    m_srmResult,
                    rayCount,
                    rseBatch,
                    betaVolumeTotal,
                    m_vertexBatchScoreSum,
                    m_volumeRayVisits,
                    m_droppedRays,
                    rngSeed,
                    controls,
                    *m_srmWorkspace);
            }
            else
            {
                BENCH_SYNC(m_queue, AccumulateForwardPhiAse);
                m_queue.enqueue(
                    frameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::forward::AccumulateForwardPhiAse{},
                        mesh,
                        rayCount,
                        rseBatch,
                        betaVolumeTotal,
                        accumulation,
                        rngSeed});
            }
        }

        void finish(ForwardPhiAseRawResult& result, float& runtime, bool downloadAccumulators = true)
        {
            result = makeForwardRawResult(m_volumeCount, m_materialVertexCount, m_batchCount);
            result.rayCount = m_accumulatedRayCount;
            result.rseBatchRayCounts = m_rseBatchRayCounts;
            result.srmStatus = m_srmResult.srmStatus;
            result.srmPasses = m_srmResult.srmPasses;
            result.srmRemainingFraction = m_srmResult.srmRemainingFraction;
            result.srmMaxIterations = m_srmResult.srmMaxIterations;
            result.srmDivergenceStreak = m_srmResult.srmDivergenceStreak;
            if(m_rayCount == 0u)
            {
                runtime = 0.0f;
                return;
            }
            if(downloadAccumulators)
            {
                alpaka::onHost::memcpy(m_queue, result.vertexBatchScoreSum, m_vertexBatchScoreSum);
                alpaka::onHost::memcpy(m_queue, result.totalRays, m_volumeRayVisits);
                alpaka::onHost::memcpy(m_queue, result.droppedRays, m_droppedRays);
            }
            alpaka::onHost::wait(m_queue);
            runtime = static_cast<float>(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - m_started).count());
        }

        void evaluate(
            hase::data::TraceView const mesh,
            ForwardPhiAseRawResult& result,
            float& runtime,
            unsigned rayCount,
            unsigned rngSeed,
            unsigned rseBatch,
            double betaVolumeTotal,
            AseTraceControls const& experiment)
        {
            begin(mesh, rayCount, rngSeed, rseBatch, betaVolumeTotal, experiment);
            finish(result, runtime);
        }

        void finalizeCellPhiAse(hase::data::TraceView const mesh, unsigned rayCount, double sourceStrengthTotal)
        {
            m_rseBatchRayCountsBuffer.toDevice(m_queue);
            hase::kernels::enqueueFinalizeForwardCellPhiAse(
                m_devBundle,
                m_queue,
                mesh,
                m_vertexBatchScoreSum,
                m_rseBatchRayCountsBuffer.toDeviceView(),
                m_lumpedMaterialVertexVolume,
                m_droppedRays,
                m_volumePhiAse,
                m_standardError,
                m_relativeStandardError,
                m_volumeDndtAse,
                rayCount,
                m_batchCount,
                sourceStrengthTotal);
            alpaka::onHost::wait(m_queue);
        }

        /**
         * @brief Upload gathered raw batches and finalize them on this device.
         *
         * This is the explicit host/device boundary after a thread or MPI
         * gather. Normalization, RSE evaluation, and ASE derivative generation
         * remain device-side.
         */
        void uploadAndFinalize(
            hase::data::TraceView const mesh,
            ForwardPhiAseRawResult const& rawResult,
            double const sourceStrengthTotal)
        {
            if(rawResult.rseBatchRayCounts.size() != m_batchCount
               || rawResult.vertexBatchScoreSum.size() != m_vertexBatchScoreSum.getExtents().product()
               || rawResult.totalRays.size() != m_volumeCount || rawResult.droppedRays.size() != m_volumeCount)
                throw std::runtime_error("gathered forward ASE result does not match the device context");
            alpaka::onHost::memcpy(m_queue, m_vertexBatchScoreSum, rawResult.vertexBatchScoreSum);
            alpaka::onHost::memcpy(m_queue, m_volumeRayVisits, rawResult.totalRays);
            alpaka::onHost::memcpy(m_queue, m_droppedRays, rawResult.droppedRays);
            m_rseBatchRayCounts = rawResult.rseBatchRayCounts;
            m_accumulatedRayCount = rawResult.rayCount;
            finalizeCellPhiAse(mesh, rawResult.rayCount, sourceStrengthTotal);
        }

        double rebuildSourceStrengthPrefix(hase::data::ResidentTrace<T_Device>& meshContainer, auto const& betaVolume)
        {
            auto mesh = meshContainer.view();
            mesh.betaVolume = std::span<double const>(betaVolume.data(), betaVolume.getExtents().x());
            if(mesh.numberOfCells == 0u)
            {
                alpaka::onHost::fill(m_queue, m_sourceStrengthTotal.toDeviceView(), 0.0, alpaka::Vec{std::size_t{1}});
            }
            else
            {
                auto const cellFrameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
                    m_devBundle.device,
                    m_devBundle.executor,
                    alpaka::Vec{mesh.numberOfCells});
                m_queue.enqueue(
                    cellFrameSpec,
                    alpaka::KernelBundle{hase::kernels::BuildSourceStrengthWeights{}, mesh, m_sourceStrengthWeights});
                auto sourceStrengthPrefix = meshContainer.sourceStrengthPrefix.toDeviceView();
                alpaka::onHost::inclusiveScan(
                    m_queue,
                    m_devBundle.executor,
                    m_sourceStrengthPrefixScanBuffer,
                    sourceStrengthPrefix,
                    m_sourceStrengthWeights);
                auto const scalarFrameSpec
                    = alpaka::onHost::getFrameSpec(m_devBundle.device, m_devBundle.executor, alpaka::Vec{1u});
                m_queue.enqueue(
                    scalarFrameSpec,
                    alpaka::KernelBundle{
                        hase::kernels::CaptureSourceStrengthTotal{},
                        mesh.numberOfCells,
                        sourceStrengthPrefix,
                        m_sourceStrengthTotal.toDeviceView()});
            }
            m_sourceStrengthTotal.toHost(m_queue);
            return m_sourceStrengthTotal.getHostView()[0u];
        }

        std::vector<double> downloadVolumeDndtAse()
        {
            std::vector<double> result(m_volumeCount, 0.0);
            alpaka::onHost::memcpy(m_queue, result, m_volumeDndtAse);
            alpaka::onHost::wait(m_queue);
            return result;
        }

        data::PhiAseResult downloadFinalizedResult(
            bool includePhiAse,
            bool includeStandardError,
            bool includeRelativeStandardError,
            bool includeTotalRays)
        {
            data::PhiAseResult result;
            if(includePhiAse)
            {
                result.phiAse.resize(m_volumeCount);
                alpaka::onHost::memcpy(m_queue, result.phiAse, m_volumePhiAse);
            }
            if(includeStandardError)
            {
                result.standardError.resize(m_volumeCount);
                alpaka::onHost::memcpy(m_queue, result.standardError, m_standardError);
            }
            if(includeRelativeStandardError)
            {
                result.relativeStandardError.resize(m_volumeCount);
                alpaka::onHost::memcpy(m_queue, result.relativeStandardError, m_relativeStandardError);
            }
            if(includeTotalRays)
            {
                result.totalRays.resize(m_volumeCount);
                result.droppedRays.resize(m_volumeCount);
                alpaka::onHost::memcpy(m_queue, result.totalRays, m_volumeRayVisits);
                alpaka::onHost::memcpy(m_queue, result.droppedRays, m_droppedRays);
            }
            alpaka::onHost::wait(m_queue);
            return result;
        }

        [[nodiscard]] auto& volumePhiAse()
        {
            return m_volumePhiAse;
        }

        [[nodiscard]] auto& volumeDndtAse()
        {
            return m_volumeDndtAse;
        }

    private:
        hase::alpakaUtils::DevBundle<T_Device, T_Exec> m_devBundle;
        T_Queue m_queue;
        std::vector<unsigned> m_rseBatchRayCounts;
        T_RseBatchRayCounts m_rseBatchRayCountsBuffer;
        std::array<double, 1u> m_sourceStrengthTotalHost;
        T_BetaVolumeTotal m_sourceStrengthTotal;
        T_DoubleBuffer m_vertexBatchScoreSum;
        T_UnsignedBuffer m_volumeRayVisits;
        T_UnsignedBuffer m_droppedRays;
        T_DoubleBuffer m_lumpedMaterialVertexVolume;
        T_FloatBuffer m_volumePhiAse;
        T_DoubleBuffer m_standardError;
        T_DoubleBuffer m_relativeStandardError;
        T_DoubleBuffer m_volumeDndtAse;
        T_DoubleBuffer m_sourceStrengthWeights;
        T_CharBuffer m_sourceStrengthPrefixScanBuffer;
        std::unique_ptr<ForwardSrmWorkspace<T_Device>> m_srmWorkspace;
        ForwardPhiAseRawResult m_srmResult;
        unsigned m_volumeCount;
        unsigned m_materialVertexCount;
        unsigned m_batchCount;
        unsigned m_rayCount = 0u;
        unsigned m_accumulatedRayCount = 0u;
        std::chrono::steady_clock::time_point m_started;
    };

} // namespace hase::core
