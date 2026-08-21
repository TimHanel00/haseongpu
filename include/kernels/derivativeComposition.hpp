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
    template<typename T>
    concept ComposeDerivativeBufferHandle = requires(T buffers) {
        buffers.betaVolume;
        buffers.phiAse;
        buffers.dndtPump;
        buffers.dndtAse;
        buffers.derivative;
    };

    struct ComposeDerivative
    {
        bool pumpEnabled = true;

        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            data::TraceView const mesh,
            auto betaVolume,
            auto phiAse,
            auto dndtPump,
            auto dndtAse,
            auto derivative) const
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

    void enqueueComposeDerivative(
        auto& devBundle,
        concepts::Queue auto const& queue,
        auto const& mesh,
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
