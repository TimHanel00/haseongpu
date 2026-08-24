/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/CVec.hpp>

#include <cstdint>

#ifndef HASE_EXACT_PUMP_CACHE
#    define HASE_EXACT_PUMP_CACHE 1
#endif

#ifndef HASE_RESERVOIR_SLOTS_PUMP
#    define HASE_RESERVOIR_SLOTS_PUMP 64
#endif

namespace hase::core::compileTimeConfig
{
    inline constexpr bool exactPumpCache = HASE_EXACT_PUMP_CACHE != 0;
    inline constexpr alpaka::concepts::CVector auto pumpReservoirSlots
        = alpaka::CVec<std::uint32_t, HASE_RESERVOIR_SLOTS_PUMP>{};

    static_assert(exactPumpCache || pumpReservoirSlots.x() > 0u);
} // namespace hase::core::compileTimeConfig
