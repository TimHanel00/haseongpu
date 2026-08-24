/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace hase::kernels::forward
{
    /**
     * @param pass Transport pass index.
     * @param rayIndex Ray index within the pass.
     * @return Unique logical random history ID for the pass and ray.
     */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t rayHistoryId(std::uint32_t const pass, std::uint32_t const rayIndex)
    {
        return (static_cast<std::uint64_t>(pass) << 32u) | rayIndex;
    }

    /**
     * @param pass Surface-reservoir pass index.
     * @return Logical random history ID in the namespace reserved for face stratification.
     */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t surfaceSamplingHistoryId(std::uint32_t const pass)
    {
        return (std::uint64_t{1u} << 63u) | pass;
    }
} // namespace hase::kernels::forward
