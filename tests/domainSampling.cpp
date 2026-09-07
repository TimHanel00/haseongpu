#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/forwardSamplingPlan.hpp>

#include <array>
#include <numeric>
#include <vector>

TEST_CASE("high contrast domain and boundary quotas exactly match the budget", "[forward][correctness][schedule]")
{
    using namespace hase::core;
    std::vector<DomainCost> const domains{
        {0u, 1u, 1u, 0u, 99.7, std::nullopt},
        {1u, 1u, 1u, 0u, 0.1, std::nullopt},
        {2u, 1u, 1u, 0u, 0.1, std::nullopt},
        {3u, 1u, 1u, 0u, 0.1, std::nullopt}};
    auto const quotas = allocateDomainRays(domains, 100u);
    CHECK(
        std::accumulate(
            quotas.begin(),
            quotas.end(),
            std::uint64_t{0u},
            [](auto total, auto const& quota) { return total + quota.rayCount; })
        == 100u);
    for(auto const& quota : quotas)
        CHECK(quota.rayCount >= 1u);
    std::array<double, 4u> const weights{99.7, 0.1, 0.1, 0.1};
    std::array<std::uint32_t, 4u> const candidates{100u, 100u, 100u, 100u};
    auto const populations = allocateBoundaryRoutePopulations(weights, 100u, candidates);
    CHECK(std::accumulate(populations.begin(), populations.end(), 0u) == 100u);
}

TEST_CASE("adaptive launch coalescing never omits a positive source", "[forward][correctness][schedule]")
{
    using namespace hase::core;
    AseTraceControls experiment;
    experiment.minRays = 10u;
    experiment.maxRays = 100u;
    ExecutionPolicy compute;
    compute.adaptiveSteps = 1u;
    std::vector<DomainQuota> const quotas{{0u, 99u, 1.0, 0.0}, {1u, 1u, 1.0, 0.0}};
    std::vector<std::uint64_t> const completed(2u, 0u);
    auto const batches = domainRayPopulationCount(quotas, 8u);
    CHECK(batches == 1u);
    unsigned increase = 0u;
    auto const launch = planForwardLaunch(experiment, compute, quotas, completed, batches, 0u, increase);
    CHECK(launch.target == 100u);
    CHECK(launch.domainCounts == std::vector<std::uint32_t>{99u, 1u});
    CHECK(increase == 1u);
}

TEST_CASE(
    "adaptive launch planner makes progress and preserves later complete batches",
    "[forward][correctness][schedule]")
{
    using namespace hase::core;
    AseTraceControls experiment;
    experiment.minRays = 2u;
    experiment.maxRays = 100u;
    ExecutionPolicy compute;
    compute.adaptiveSteps = 100u;
    std::vector<DomainQuota> const quotas{{0u, 75u, 1.0, 0.0}, {1u, 25u, 1.0, 0.0}};
    std::vector<std::uint64_t> completed(2u, 0u);
    unsigned target = 0u;
    for(unsigned increase = 0u; target < experiment.maxRays; ++increase)
    {
        REQUIRE(increase <= compute.adaptiveSteps);
        auto const launch = planForwardLaunch(experiment, compute, quotas, completed, 8u, target, increase);
        CHECK(launch.target > target);
        for(unsigned domain = 0u; domain < 2u; ++domain)
        {
            CHECK(launch.domainCounts[domain] >= 8u);
            completed[domain] += launch.domainCounts[domain];
        }
        target = launch.target;
    }
    CHECK(completed == std::vector<std::uint64_t>{75u, 25u});
    compute.adaptiveSteps = 0u;
    std::vector<DomainQuota> const smallQuotas{{0u, 1u, 1.0, 0.0}, {1u, 1u, 1.0, 0.0}};
    unsigned increase = 0u;
    auto const launch
        = planForwardLaunch(experiment, compute, smallQuotas, std::vector<std::uint64_t>(2u), 1u, 0u, increase);
    CHECK(launch.target == 2u);
    CHECK_THROWS_AS(
        planForwardLaunch(experiment, compute, smallQuotas, std::vector<std::uint64_t>{1u, 1u}, 1u, 2u, increase),
        std::runtime_error);
}
