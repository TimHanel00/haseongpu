/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <core/Runtime.hpp>
#include <data/TraceData.hpp>
#include <random/random.hpp>

namespace hase::core
{
    /**
     * @brief Evaluate ASE-induced population change for one cell.
     * @param mesh Prepared host trace containing the cell excitation.
     * @param sigmaA Absorption cross section at the evaluated wavelength.
     * @param sigmaE Emission cross section at the evaluated wavelength.
     * @param phiAse Cell ASE photon flux.
     * @param volume Cell index in prepared order.
     * @return Signed ASE population-rate contribution for the cell.
     */
    inline double calcVolumeDndtAse(
        hase::data::TraceData const& mesh,
        double const sigmaA,
        double const sigmaE,
        float const phiAse,
        unsigned const volume)
    {
        double const gainPerDensity = mesh.betaVolume[volume] * (sigmaE + sigmaA) - sigmaA;
        return gainPerDensity * phiAse;
    }

    /**
     * @param compute Execution policy containing an optional fixed seed.
     * @return Configured seed, or the process seed when it is unspecified.
     */
    inline unsigned baseRngSeed(ExecutionPolicy const& compute)
    {
        if(compute.rngSeed == ExecutionPolicy::unspecifiedRngSeed)
            return random::SeedGenerator::get().getSeed();
        return compute.rngSeed;
    }
} // namespace hase::core
