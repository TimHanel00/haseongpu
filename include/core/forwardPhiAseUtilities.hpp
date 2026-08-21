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

    inline unsigned baseRngSeed(ExecutionPolicy const& compute)
    {
        if(compute.rngSeed == ExecutionPolicy::unspecifiedRngSeed)
            return random::SeedGenerator::get().getSeed();
        return compute.rngSeed;
    }
} // namespace hase::core
