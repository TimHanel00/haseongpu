/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

namespace hase::kernels
{
    struct AddScaled
    {
        double scale = 1.0;

        ALPAKA_FN_ACC constexpr auto operator()(
            alpaka::concepts::Simd auto const& base,
            alpaka::concepts::Simd auto const& slope) const
        {
            return base + scale * slope;
        }
    };

    struct CombineHeun
    {
        double timeStep = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(
            alpaka::concepts::Simd auto const& base,
            alpaka::concepts::Simd auto const& first,
            alpaka::concepts::Simd auto const& second) const
        {
            return base + 0.5 * timeStep * (first + second);
        }
    };

    struct CombineRungeKutta4
    {
        double timeStep = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(
            alpaka::concepts::Simd auto const& base,
            alpaka::concepts::Simd auto const& k1,
            alpaka::concepts::Simd auto const& k2,
            alpaka::concepts::Simd auto const& k3,
            alpaka::concepts::Simd auto const& k4) const
        {
            return base + (timeStep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    };

    struct ExponentialEulerUpdate
    {
        double timeStep = 0.0;
        double tau = 1.0;

        ALPAKA_FN_ACC auto operator()(
            alpaka::concepts::Simd auto const& betaVolume,
            alpaka::concepts::Simd auto const& dndtPump,
            alpaka::concepts::Simd auto const& dndtAse) const
        {
            double const decay = alpaka::math::exp(-timeStep / tau);
            auto const source = dndtPump - dndtAse;
            return tau * source * (1.0 - decay) + betaVolume * decay;
        }
    };

    struct ClipBeta
    {
        ALPAKA_FN_ACC constexpr auto operator()(alpaka::concepts::Simd auto const& betaVolume) const
        {
            auto result = betaVolume;
            alpaka::where((result <= 0.0) || (result != result), result) = 0.0;
            alpaka::where(result > 1.0, result) = 1.0;
            return result;
        }
    };

} // namespace hase::kernels
