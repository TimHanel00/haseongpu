#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/forwardSamplingPlan.hpp>
#include <kernels/forward/rayPopulationStatistics.hpp>

#include <cmath>
#include <limits>

TEST_CASE("unequal source quotas have no spurious batch variance", "[forward][correctness][rse]")
{
    using namespace hase::core;
    hase::kernels::forward::RayPopulationStatistics statistics;
    for(unsigned batch = 0u; batch < 8u; ++batch)
    {
        unsigned const first = domainPopulationRayCount(99u, batch, 8u);
        unsigned const second = domainPopulationRayCount(8u, batch, 8u);
        unsigned const count = first + second;
        double const score = first * domainPopulationSourceWeight(1.0, 2.0, first, count)
                             + second * domainPopulationSourceWeight(1.0, 2.0, second, count);
        statistics.add(score, count);
    }
    auto const result = statistics.finalize(2.0, false);
    CHECK(result.value == Catch::Approx(2.0));
    CHECK(result.relativeStandardError == Catch::Approx(0.0).margin(1.0e-15));
    CHECK_THROWS_AS(domainPopulationSourceWeight(1.0, 2.0, 0u, 8u), std::invalid_argument);
}

TEST_CASE("field and RSE use the same replicate mean", "[forward][correctness][rse]")
{
    hase::kernels::forward::RayPopulationStatistics statistics;
    statistics.add(2.0, 2u);
    statistics.add(3.0, 1u);
    auto const result = statistics.finalize(1.0, false);
    CHECK(result.value == Catch::Approx(2.0));
    CHECK(result.standardError == Catch::Approx(1.0));
    CHECK(result.relativeStandardError == Catch::Approx(0.5));
}

TEST_CASE("invalid or insufficient tallies cannot claim precise ASE", "[forward][correctness][rse]")
{
    using hase::kernels::forward::RayPopulationStatistics;
    RayPopulationStatistics statistics;
    statistics.add(1.0, 1u);
    CHECK(statistics.finalize(1.0, false).relativeStandardError == std::numeric_limits<double>::max());
    statistics.add(1.0, 1u);
    CHECK(statistics.finalize(1.0, true).relativeStandardError == std::numeric_limits<double>::max());
    CHECK_FALSE(std::isfinite(statistics.finalize(1.0e39, false).value));
    CHECK(statistics.finalize(1.0e39, false).relativeStandardError == std::numeric_limits<double>::max());
    statistics.add(std::numeric_limits<double>::infinity(), 1u);
    CHECK_FALSE(std::isfinite(statistics.finalize(1.0, false).value));
}
