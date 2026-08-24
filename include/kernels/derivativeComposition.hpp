/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <concepts/concepts.hpp>
#include <data/TraceData.hpp>

namespace hase::kernels
{
    /** @brief Buffer bundle required to compose one population derivative field. */
    template<typename T>
    concept ComposeDerivativeBufferHandle = requires(T buffers) {
        buffers.betaVolume;
        buffers.phiAse;
        buffers.dndtPump;
        buffers.dndtAse;
        buffers.derivative;
        requires alpaka::concepts::IBuffer<decltype(buffers.betaVolume), double>;
        requires alpaka::concepts::IBuffer<decltype(buffers.phiAse), float>;
        requires alpaka::concepts::IBuffer<decltype(buffers.dndtPump), double>;
        requires alpaka::concepts::IBuffer<decltype(buffers.dndtAse), double>;
        requires alpaka::concepts::IBuffer<decltype(buffers.derivative), double>;
    };

    /** @brief Device operation composing pump, ASE, and fluorescence terms per cell. */
    struct ComposeDerivative
    {
        bool pumpEnabled = true;

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            data::TraceView const mesh,
            alpaka::concepts::IView<double> auto betaVolume,
            alpaka::concepts::IView<float> auto phiAse,
            alpaka::concepts::IView<double> auto dndtPump,
            alpaka::concepts::IView<double> auto dndtAse,
            alpaka::concepts::IView<double> auto derivative) const
        {
            for(auto [cell] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{mesh.numberOfCells}))
            {
                if(!mesh.isActive(cell))
                {
                    dndtPump[cell] = 0.0;
                    dndtAse[cell] = 0.0;
                    derivative[cell] = 0.0;
                    continue;
                }

                double const pumpTerm = pumpEnabled ? dndtPump[cell] : 0.0;
                unsigned const material = mesh.getMaterialId(cell);
                double const gainPerDensity
                    = betaVolume[cell] * (mesh.materialPeakEmission[material] + mesh.materialPeakAbsorption[material])
                      - mesh.materialPeakAbsorption[material];
                double const aseTerm = gainPerDensity * static_cast<double>(phiAse[cell]);
                if(!pumpEnabled)
                    dndtPump[cell] = 0.0;
                dndtAse[cell] = aseTerm;
                derivative[cell] = pumpTerm - aseTerm - betaVolume[cell] / mesh.fluorescenceLifetime(cell);
            }
        }
    };

    /**
     * @brief Enqueue population-derivative composition for every prepared cell.
     * @param devBundle Device and executor used to build the launch frame.
     * @param queue Queue receiving the kernel.
     * @param mesh Device-resident trace view supplying material properties.
     * @param pumpEnabled Whether the existing pump-rate field contributes.
     * @param buffers Beta, PhiASE, pump/ASE-rate, and output derivative buffers.
     */
    template<alpaka::onHost::concepts::Device T_Device, alpaka::concepts::Executor T_Executor>
    void enqueueComposeDerivative(
        alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
        concepts::Queue auto const& queue,
        data::TraceView const mesh,
        bool pumpEnabled,
        ComposeDerivativeBufferHandle auto& buffers)
    {
        auto frameSpec = alpakaUtils::getFrameSpec<uint32_t>(
            devBundle.device,
            devBundle.executor,
            alpaka::Vec{mesh.numberOfCells});
        queue.enqueue(
            frameSpec,
            alpaka::KernelBundle{
                ComposeDerivative{pumpEnabled},
                mesh,
                buffers.betaVolume,
                buffers.phiAse,
                buffers.dndtPump,
                buffers.dndtAse,
                buffers.derivative});
    }

} // namespace hase::kernels
