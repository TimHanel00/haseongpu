#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <kernels/forward/accumulation.hpp>
#include <kernels/forward/volumeSampling.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

TEST_CASE("spectral stratum permutation is bijective for arbitrary populations", "[forward][sampling]")
{
    for(std::uint32_t const count : {1u, 2u, 3u, 7u, 16u, 25u, 257u, 1000u})
        for(std::uint32_t const seed : {0u, 28u, 1234u})
        {
            std::vector<std::uint32_t> visits(count, 0u);
            for(std::uint32_t ray = 0u; ray < count; ++ray)
            {
                auto const index = hase::random::permuteStratum(ray, count, seed);
                REQUIRE(index < count);
                ++visits[index];
            }
            CHECK(std::ranges::all_of(visits, [](auto const count) { return count == 1u; }));
        }
}

TEST_CASE("joint source and wavelength sampling improves with ray count", "[forward][sampling][rse]")
{
    using namespace hase::kernels::forward;
    std::array<double, 2u> empiricalVariances{};
    std::uint32_t populationIndex = 0u;
    for(std::uint32_t const count : {128u, 2048u})
    {
        double meanSum = 0.0;
        double meanSquareSum = 0.0;
        double estimatedVarianceSum = 0.0;
        constexpr std::uint32_t seeds = 256u;
        for(std::uint32_t seed = 0u; seed < seeds; ++seed)
        {
            std::array<double, 8u> batchMeans{};
            for(std::uint32_t batch = 0u; batch < batchMeans.size(); ++batch)
            {
                auto const shift = rseBatchSourceStratificationOffset(seed, batch);
                auto const phase = rseBatchSpectrumStratificationPhase(seed, batch, 2u);
                auto const key = rseBatchSpectrumPermutationSeed(seed, batch);
                for(std::uint32_t ray = 0u; ray < count; ++ray)
                {
                    auto const source = stratifiedUnitInterval(ray, count, shift) < 0.5 ? 0u : 1u;
                    auto const wavelength = stratifiedSpectrumIndex(2u, ray, count, phase, key);
                    // Independent equal-probability source and wavelength labels give E[score] = 1.5.
                    batchMeans[batch] += 1.0 + static_cast<double>(source == wavelength);
                }
                batchMeans[batch] /= count;
            }
            double const mean = std::accumulate(batchMeans.begin(), batchMeans.end(), 0.0) / batchMeans.size();
            double variance = 0.0;
            for(double const batchMean : batchMeans)
                variance += (batchMean - mean) * (batchMean - mean);
            variance /= batchMeans.size() * (batchMeans.size() - 1u);
            meanSum += mean;
            meanSquareSum += mean * mean;
            estimatedVarianceSum += variance;
            if(seed == 28u)
            {
                CHECK(std::abs(mean - 1.5) < 0.05);
                CHECK(variance > 0.0);
            }
        }
        double const ensembleMean = meanSum / seeds;
        double const empiricalVariance = (meanSquareSum - meanSum * ensembleMean) / (seeds - 1u);
        CHECK(std::abs(ensembleMean - 1.5) < 0.005);
        CHECK(estimatedVarianceSum / seeds / empiricalVariance > 0.7);
        CHECK(estimatedVarianceSum / seeds / empiricalVariance < 1.3);
        empiricalVariances[populationIndex++] = empiricalVariance;
    }
    CHECK(empiricalVariances[1] < empiricalVariances[0] / 8.0);
}
