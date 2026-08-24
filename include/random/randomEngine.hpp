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

namespace hase::random
{
    /**
     * @brief Given a base seed and a logical history ID, create a deterministic, non-overlapping random stream.
     * @param baseSeed Seed shared by the complete sampling operation.
     * @param historyId Logical history identifier within that operation.
     * @return Random engine positioned at the start of the selected history stream.
     */
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto makeRandomEngine(
        std::uint64_t const baseSeed,
        std::uint64_t const historyId)
    {
        return alpaka::rand::engine::Philox4x32x10{baseSeed, historyId};
    }
} // namespace hase::random
