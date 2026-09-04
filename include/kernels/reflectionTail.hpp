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
    /** @brief Add a scaled final reflected-pass increment to an accumulated score. */
    struct CompleteReflectionTail
    {
        double factor = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(
            alpaka::concepts::Simd auto const& accumulated,
            alpaka::concepts::Simd auto const& beforeFinalPass) const
        {
            return accumulated + factor * (accumulated - beforeFinalPass);
        }
    };
} // namespace hase::kernels
