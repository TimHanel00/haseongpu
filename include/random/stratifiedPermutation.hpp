/**
 * Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace hase::random
{
    /**
     * @brief Permute a finite stratum index without allocating a shuffle buffer.
     *
     * A keyed Feistel permutation acts on the smallest enclosing even-bit domain.
     * Cycle walking restricts that bijection to [0, count), so every stratum occurs
     * exactly once. Use an independent key for each sampling dimension and batch.
     */
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr std::uint32_t permuteStratum(
        std::uint32_t index,
        std::uint32_t const count,
        std::uint32_t const seed)
    {
        if(count <= 1u)
            return 0u;
        std::uint32_t halfBits = 1u;
        while(halfBits < 16u && (std::uint64_t{1u} << (2u * halfBits)) < count)
            ++halfBits;
        std::uint32_t const mask = (std::uint32_t{1u} << halfBits) - 1u;
        do
        {
            std::uint32_t left = index >> halfBits;
            std::uint32_t right = index & mask;
            for(std::uint32_t round = 0u; round < 8u; ++round)
            {
                std::uint32_t mixed = right ^ seed ^ (0x9e37'79b9u * (round + 1u));
                mixed ^= mixed >> 16u;
                mixed *= 0x7feb'352du;
                mixed ^= mixed >> 15u;
                mixed *= 0x846c'a68bu;
                mixed ^= mixed >> 16u;
                auto const next = left ^ (mixed & mask);
                left = right;
                right = next;
            }
            index = (left << halfBits) | right;
        } while(index >= count);
        return index;
    }
} // namespace hase::random
