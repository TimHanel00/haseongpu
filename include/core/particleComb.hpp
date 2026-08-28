#pragma once

#include <alpaka/alpaka.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <concepts/concepts.hpp>
#include <kernels/forward/particleCombing.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

namespace hase::core
{
    /** @brief Persistent device-only CDF and selection workspace for systematic particle combing. */
    template<alpaka::onHost::concepts::Device T_Device>
    class ParticleCombWorkspace
    {
        using T_DoubleBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint32_t>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_ByteBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<char>(std::declval<T_Device&>(), std::size_t{1u}));

    public:
        ParticleCombWorkspace(
            T_Device& device,
            std::uint32_t const maxCandidateCount,
            std::uint32_t const maxOutputCount)
            : cdf(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(maxCandidateCount)))
            , totalWeight(alpaka::onHost::alloc<double>(device, std::size_t{1u}))
            , systematicOffset(alpaka::onHost::alloc<double>(device, std::size_t{1u}))
            , filteredWeights(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(maxCandidateCount)))
            , selectedCandidates(
                  alpaka::onHost::alloc<std::uint32_t>(device, static_cast<std::size_t>(maxOutputCount)))
            , selectedWeights(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(maxOutputCount)))
            , scanBuffer(
                  alpaka::onHost::alloc<char>(
                      device,
                      alpaka::onHost::getScanBufferSize<double>(
                          alpaka::Vec{static_cast<std::size_t>(maxCandidateCount)})))
            , m_maxCandidateCount(maxCandidateCount)
            , m_maxOutputCount(maxOutputCount)
        {
        }

        /** @brief Enqueue scan, total capture, random offset, and candidate selection without waiting. */
        template<alpaka::concepts::Executor T_Executor>
        void enqueue(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            alpaka::concepts::IView<double> auto candidateWeights,
            std::uint32_t const candidateCount,
            std::uint32_t const outputCount,
            std::uint32_t const seed,
            std::uint64_t const historyId)
        {
            if(candidateCount > m_maxCandidateCount || outputCount > m_maxOutputCount)
                throw std::out_of_range("particle-comb workspace capacity exceeded");
            auto const extent = alpaka::Vec{static_cast<std::size_t>(candidateCount)};
            auto weightView = candidateWeights.getSubView(extent);
            enqueueSelection(devBundle, queue, weightView, candidateCount, 0u, outputCount, seed, historyId);
        }

        /** @brief Enqueue systematic combing for one target domain into a selected-output segment. */
        template<alpaka::concepts::Executor T_Executor>
        void enqueueDomain(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            alpaka::concepts::IView<double> auto candidateWeights,
            alpaka::concepts::IView<std::uint32_t> auto candidateDomains,
            std::uint32_t const candidateCount,
            std::uint32_t const domain,
            std::uint32_t const outputOffset,
            std::uint32_t const outputCount,
            std::uint32_t const seed,
            std::uint64_t const historyId)
        {
            if(candidateCount > m_maxCandidateCount || outputOffset > m_maxOutputCount
               || outputCount > m_maxOutputCount - outputOffset)
                throw std::out_of_range("particle-comb domain segment exceeds workspace capacity");
            auto const candidateExtent = alpaka::Vec{static_cast<std::size_t>(candidateCount)};
            auto const candidateFrame = hase::alpakaUtils::getFrameSpec<std::uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{candidateCount});
            auto filteredView = filteredWeights.getView().getSubView(candidateExtent);
            queue.enqueue(
                candidateFrame,
                alpaka::KernelBundle{
                    kernels::forward::FilterParticleCombDomainWeights{},
                    candidateWeights.getSubView(candidateExtent),
                    candidateDomains.getSubView(candidateExtent),
                    filteredView,
                    candidateCount,
                    domain});
            enqueueSelection(
                devBundle,
                queue,
                filteredView,
                candidateCount,
                outputOffset,
                outputCount,
                seed,
                historyId);
        }

        /** @brief Enqueue one domain segment when candidates are mesh faces. */
        template<alpaka::concepts::Executor T_Executor>
        void enqueueFaceDomain(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            alpaka::concepts::IView<double> auto faceWeights,
            std::span<std::uint32_t const> const cellDomains,
            std::uint32_t const faceCount,
            std::uint32_t const facesPerCell,
            std::uint32_t const domain,
            std::uint32_t const outputOffset,
            std::uint32_t const outputCount,
            std::uint32_t const seed,
            std::uint64_t const historyId)
        {
            if(faceCount > m_maxCandidateCount || outputOffset > m_maxOutputCount
               || outputCount > m_maxOutputCount - outputOffset)
                throw std::out_of_range("particle-comb face-domain segment exceeds workspace capacity");
            auto const candidateExtent = alpaka::Vec{static_cast<std::size_t>(faceCount)};
            auto const candidateFrame = hase::alpakaUtils::getFrameSpec<std::uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{faceCount});
            auto filteredView = filteredWeights.getView().getSubView(candidateExtent);
            queue.enqueue(
                candidateFrame,
                alpaka::KernelBundle{
                    kernels::forward::FilterParticleCombFaceDomainWeights{},
                    faceWeights.getSubView(candidateExtent),
                    cellDomains,
                    filteredView,
                    faceCount,
                    facesPerCell,
                    domain});
            enqueueSelection(devBundle, queue, filteredView, faceCount, outputOffset, outputCount, seed, historyId);
        }

        /** @brief Enqueue only the candidate scan and total capture used for convergence. */
        template<alpaka::concepts::Executor T_Executor>
        void enqueueTotal(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            alpaka::concepts::IView<double> auto candidateWeights,
            std::uint32_t const candidateCount)
        {
            if(candidateCount > m_maxCandidateCount)
                throw std::out_of_range("particle-comb candidate count exceeds workspace capacity");
            auto const extent = alpaka::Vec{static_cast<std::size_t>(candidateCount)};
            auto cdfView = cdf.getView().getSubView(extent);
            auto weightView = candidateWeights.getSubView(extent);
            alpaka::onHost::inclusiveScan(queue, devBundle.executor, scanBuffer, cdfView, weightView);
            auto const scalarFrame = alpaka::onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u}, devBundle.executor};
            queue.enqueue(
                scalarFrame,
                alpaka::KernelBundle{
                    kernels::forward::CaptureParticleCombTotalWeight{},
                    cdfView,
                    totalWeight.getView(),
                    candidateCount});
        }

        [[nodiscard]] auto selectedView(std::uint32_t const count)
        {
            return selectedCandidates.getView().getSubView(alpaka::Vec{static_cast<std::size_t>(count)});
        }

        [[nodiscard]] auto totalWeightView()
        {
            return totalWeight.getView();
        }

        [[nodiscard]] auto selectedWeightsView(std::uint32_t const count)
        {
            return selectedWeights.getView().getSubView(alpaka::Vec{static_cast<std::size_t>(count)});
        }

        T_DoubleBuffer cdf;
        T_DoubleBuffer totalWeight;
        T_DoubleBuffer systematicOffset;
        T_DoubleBuffer filteredWeights;
        T_UnsignedBuffer selectedCandidates;
        T_DoubleBuffer selectedWeights;
        T_ByteBuffer scanBuffer;

    private:
        template<alpaka::concepts::Executor T_Executor>
        void enqueueSelection(
            alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
            concepts::Queue auto const& queue,
            alpaka::concepts::IView<double> auto candidateWeights,
            std::uint32_t const candidateCount,
            std::uint32_t const outputOffset,
            std::uint32_t const outputCount,
            std::uint32_t const seed,
            std::uint64_t const historyId)
        {
            auto const extent = alpaka::Vec{static_cast<std::size_t>(candidateCount)};
            auto cdfView = cdf.getView().getSubView(extent);
            alpaka::onHost::inclusiveScan(queue, devBundle.executor, scanBuffer, cdfView, candidateWeights);
            auto const scalarFrame = alpaka::onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u}, devBundle.executor};
            queue.enqueue(
                scalarFrame,
                alpaka::KernelBundle{
                    kernels::forward::CaptureParticleCombTotalWeight{},
                    cdfView,
                    totalWeight.getView(),
                    candidateCount});
            queue.enqueue(
                scalarFrame,
                alpaka::KernelBundle{
                    kernels::forward::GenerateParticleCombOffset{},
                    totalWeight.getView(),
                    systematicOffset.getView(),
                    outputCount,
                    seed,
                    historyId});
            auto const outputFrame = alpaka::onHost::FrameSpec{
                alpaka::Vec{(outputCount + 127u) / 128u},
                alpaka::Vec{128u},
                devBundle.executor};
            queue.enqueue(
                outputFrame,
                alpaka::KernelBundle{
                    kernels::forward::SelectParticleCombCandidates{},
                    cdfView,
                    totalWeight.getView(),
                    systematicOffset.getView(),
                    selectedCandidates.getView(),
                    candidateCount,
                    outputCount,
                    outputOffset});
            queue.enqueue(
                outputFrame,
                alpaka::KernelBundle{
                    kernels::forward::AssignParticleCombSelectedWeights{},
                    totalWeight.getView(),
                    selectedWeights.getView(),
                    outputCount,
                    outputOffset});
        }

        std::uint32_t m_maxCandidateCount;
        std::uint32_t m_maxOutputCount;
    };
} // namespace hase::core
