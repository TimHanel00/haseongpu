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
#include <alpakaUtils/utils.hpp>
#include <concepts/concepts.hpp>
#include <core/mesh.hpp>

#include <cmath>

namespace hase::kernels
{
    template<typename T>
    concept ComposeDerivativeBufferHandle = requires(T buffers) {
        buffers.betaVolume;
        buffers.dndtPump;
        buffers.dndtAse;
        buffers.derivative;
    };

    struct ComposeDerivative
    {
        double tau = 1.0;
        bool pumpEnabled = true;

        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            hase::core::DeviceMeshView const mesh,
            auto betaVolume,
            auto dndtPump,
            auto dndtAse,
            auto derivative) const
        {
            for(auto [cell] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{mesh.numberOfCells}))
            {
                if(mesh.getCellType(cell) == mesh.claddingNumber)
                {
                    dndtPump[cell] = 0.0;
                    dndtAse[cell] = 0.0;
                    derivative[cell] = 0.0;
                    continue;
                }

                double const pumpTerm = pumpEnabled ? dndtPump[cell] : 0.0;
                if(!pumpEnabled)
                    dndtPump[cell] = 0.0;
                derivative[cell] = pumpTerm - dndtAse[cell] - betaVolume[cell] / tau;
            }
        }
    };

    void enqueueComposeDerivative(
        auto& devBundle,
        hase::concepts::Queue auto const& queue,
        auto const& mesh,
        double tau,
        bool pumpEnabled,
        ComposeDerivativeBufferHandle auto& buffers)
    {
        auto frameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
            devBundle.device,
            devBundle.executor,
            alpaka::Vec{mesh.numberOfCells});
        queue.enqueue(
            frameSpec,
            alpaka::KernelBundle{
                ComposeDerivative{tau, pumpEnabled},
                mesh,
                buffers.betaVolume,
                buffers.dndtPump,
                buffers.dndtAse,
                buffers.derivative});
    }

} // namespace hase::kernels
