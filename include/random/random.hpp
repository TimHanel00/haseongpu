/* Copyright 2026 Tim Hanel
 */
#pragma once
#include <cstdint>
#include <random>

namespace hase::random
{
    namespace internal
    {
        inline constexpr std::uint64_t boostHashMix(std::uint64_t x)
        {
            std::uint64_t const m = 0xe9'846a'fb1a'615dull;
            x ^= x >> 32;
            x *= m;
            x ^= x >> 32;
            x *= m;
            x ^= x >> 28;
            return x;
        }

        inline constexpr std::uint32_t mixSeed(std::uint32_t seed, std::uint32_t value)
        {
            // boost::hash_combine
            return boostHashMix(seed + 0x9e37'79b9 + value);
        }
    } // namespace internal

    /** @brief Process-wide seed provider with an optional caller-selected seed. */
    struct SeedGenerator
    {
        /** @brief Initialize the provider from `std::random_device`. */
        explicit SeedGenerator() : fixedSeed(std::random_device{}())
        {
        }

        SeedGenerator(SeedGenerator const&) = delete;
        SeedGenerator& operator=(SeedGenerator const&) = delete;

        SeedGenerator(SeedGenerator&&) = delete;
        SeedGenerator& operator=(SeedGenerator&&) = delete;

        /** @param seed Fixed base seed returned by subsequent `getSeed()` calls. */
        void updateSeed(unsigned const seed)
        {
            fixedSeed = seed;
        }

        /** @return Current process-wide base seed. */
        [[nodiscard]] unsigned getSeed() const
        {
            return fixedSeed;
        }

        /** @return Process-wide seed-provider singleton. */
        static SeedGenerator& get()
        {
            static SeedGenerator provider{};
            return provider;
        }

    private:
        unsigned fixedSeed;
    };

    /**
     * @param base Run-level base seed.
     * @param rank Worker rank.
     * @param deviceIndex Device index owned by the worker.
     * @return Deterministically mixed worker seed.
     */
    inline constexpr std::uint32_t seedForWorker(std::uint32_t base, std::uint32_t rank, std::uint32_t deviceIndex)
    {
        return internal::mixSeed(internal::mixSeed(base, rank), deviceIndex);
    }

    /**
     * @param base Run-level base seed.
     * @param launch Zero-based adaptive launch index.
     * @return `base` for the first launch and a deterministic mixed seed thereafter.
     */
    inline constexpr std::uint32_t seedForAdaptiveLaunch(std::uint32_t base, std::uint32_t launch)
    {
        return launch == 0u ? base : internal::mixSeed(base, launch);
    }

    /**
     * @param base Seed identifying the stratified sample set.
     * @return Deterministic offset in the half-open unit interval `[0, 1)`.
     */
    inline constexpr double stratifiedUnitOffset(std::uint32_t const base)
    {
        return static_cast<double>(internal::mixSeed(base, 0x7d3a'9f21u)) / 4294967296.0;
    }

    /**
     * @param base Seed identifying the stratified sample set.
     * @param spectrumSize Number of discrete spectrum entries.
     * @return Deterministic phase in `[0, spectrumSize)`, or zero for an empty spectrum.
     */
    inline constexpr std::uint32_t stratifiedSpectrumPhase(std::uint32_t const base, std::uint32_t const spectrumSize)
    {
        return spectrumSize == 0u ? 0u : internal::mixSeed(base, 0x6ca4'c37du) % spectrumSize;
    }

} // namespace hase::random
