#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/domainSchedule.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

TEST_CASE("domain ray quotas reserve manual counts and distribute the remainder", "[forward][domain][schedule]")
{
    using namespace hase::core;
    std::vector<DomainCost> domains{
        {0u, 100u, 1000u, 20u, 1.0, 20u},
        {1u, 200u, 2000u, 30u, 3.0, std::nullopt},
        {2u, 50u, 500u, 10u, 1.0, std::nullopt}};

    auto const quotas = allocateDomainRays(domains, 100u);

    REQUIRE(quotas.size() == 3u);
    CHECK(quotas[0].rayCount == 20u);
    CHECK(quotas[1].rayCount == 60u);
    CHECK(quotas[2].rayCount == 20u);
    CHECK(quotas[0].importanceWeight == Catch::Approx(5.0));
    CHECK(quotas[1].importanceWeight == Catch::Approx(5.0));
    CHECK(quotas[2].importanceWeight == Catch::Approx(5.0));
}

TEST_CASE("domain ray quotas keep zero-source launches defined", "[forward][domain][schedule]")
{
    using namespace hase::core;
    auto const automatic = allocateDomainRays(
        {DomainCost{0u, 1u, 1u, 0u, 0.0, std::nullopt}, DomainCost{1u, 3u, 1u, 0u, 0.0, std::nullopt}},
        8u);
    REQUIRE(automatic.size() == 2u);
    CHECK(automatic[0u].rayCount == 2u);
    CHECK(automatic[1u].rayCount == 6u);
    CHECK(automatic[0u].importanceWeight == 0.0);
    CHECK(automatic[1u].importanceWeight == 0.0);

    auto const manual = allocateDomainRays({DomainCost{0u, 1u, 1u, 0u, 0.0, 1u}}, 1u);
    REQUIRE(manual.size() == 1u);
    CHECK(manual[0u].rayCount == 1u);
    CHECK(manual[0u].importanceWeight == 0.0);
}

TEST_CASE("domain ray quotas reject undersampled source domains", "[forward][domain][schedule]")
{
    using namespace hase::core;
    CHECK_THROWS_AS(
        allocateDomainRays(
            {DomainCost{0u, 1u, 1u, 0u, 1.0, std::nullopt}, DomainCost{1u, 1u, 1u, 0u, 1.0, std::nullopt}},
            1u),
        std::invalid_argument);
}

TEST_CASE("domain populations preserve worker and domain totals", "[forward][domain][schedule]")
{
    auto const populations = hase::core::distributeDomainPopulations(
        std::vector<std::uint32_t>{5u, 3u, 2u},
        std::vector<std::uint32_t>{4u, 3u, 3u});

    REQUIRE(populations.size() == 3u);
    for(std::size_t worker = 0u; worker < populations.size(); ++worker)
    {
        auto const workerTotal
            = std::accumulate(populations[worker].begin(), populations[worker].end(), std::uint32_t{0u});
        CHECK(workerTotal == std::vector<std::uint32_t>{4u, 3u, 3u}[worker]);
    }
    for(std::size_t domain = 0u; domain < 3u; ++domain)
    {
        std::uint32_t domainTotal = 0u;
        for(auto const& worker : populations)
            domainTotal += worker[domain];
        CHECK(domainTotal == std::vector<std::uint32_t>{5u, 3u, 2u}[domain]);
    }
}

TEST_CASE("adaptive domain launches consume remaining quotas monotonically", "[forward][domain][schedule]")
{
    using namespace hase::core;
    std::vector<DomainQuota> const quotas{{0u, 5u, 5.0, 1.0}, {1u, 3u, 3.0, 1.0}, {2u, 2u, 2.0, 1.0}};
    std::vector<std::uint64_t> completed(quotas.size(), 0u);

    for(auto const launchSize : std::vector<std::uint64_t>{3u, 3u, 4u})
    {
        auto const launch = allocateDomainLaunchCounts(quotas, completed, launchSize);
        CHECK(std::accumulate(launch.begin(), launch.end(), std::uint32_t{0u}) == launchSize);
        for(std::size_t domain = 0u; domain < launch.size(); ++domain)
        {
            completed[domain] += launch[domain];
            CHECK(completed[domain] <= quotas[domain].rayCount);
        }
    }
    CHECK(completed == std::vector<std::uint64_t>{5u, 3u, 2u});
}

TEST_CASE("passive domains retain a boundary relaunch slot when capacity permits", "[forward][domain][schedule]")
{
    auto const populations = hase::core::reservePassiveDomainPopulationSlots(std::vector<std::uint32_t>{5u, 0u, 0u});
    CHECK(populations == std::vector<std::uint32_t>{3u, 1u, 1u});
    CHECK(
        hase::core::reservePassiveDomainPopulationSlots(std::vector<std::uint32_t>{1u, 0u})
        == std::vector<std::uint32_t>{1u, 0u});
}

TEST_CASE("domain schedule owns every batch once and exposes resident domains", "[forward][domain][schedule]")
{
    using namespace hase::core;
    std::vector<WorkerDescriptor> workers{
        {0u, 0u, 0u, 1.0, 10000u},
        {1u, 0u, 1u, 1.0, 10000u},
        {2u, 1u, 0u, 1.0, 10000u}};
    std::vector<DomainCost> domains{
        {0u, 1000u, 1000u, 100u, 3.0, std::nullopt},
        {1u, 1000u, 1000u, 100u, 1.0, std::nullopt}};
    auto const quotas = allocateDomainRays(domains, 80u);
    std::vector<hase::data::AseDomainInterface> interfaces{{0u, 1u, 0u, 0u, 0u, 0u, 1u, 0.0, 1.0, 1.0}};

    auto const schedule = makeDomainSchedule(workers, domains, quotas, 4u, interfaces);

    REQUIRE(schedule.assignments().size() == 8u);
    for(std::uint32_t domain = 0u; domain < 2u; ++domain)
        for(std::uint32_t batch = 0u; batch < 4u; ++batch)
            CHECK(schedule.owner({domain, batch}) < workers.size());
    std::uint64_t scheduledRays = 0u;
    for(auto const& assignment : schedule.assignments())
        scheduledRays += assignment.rayCount;
    CHECK(scheduledRays == 80u);

    std::vector<hase::data::DomainId> allRequired;
    for(auto const& worker : workers)
    {
        auto required = schedule.requiredDomains(worker.id);
        allRequired.insert(allRequired.end(), required.begin(), required.end());
    }
    CHECK(std::ranges::find(allRequired, hase::data::DomainId{0u}) != allRequired.end());
    CHECK(std::ranges::find(allRequired, hase::data::DomainId{1u}) != allRequired.end());
}

TEST_CASE("domain scheduling statistics are derived from prepared shards", "[forward][domain][schedule]")
{
    hase::data::AseDomain domain;
    domain.id = 3u;
    domain.trace.numberOfCells = 2u;
    domain.trace.cellFaceBoundaries = {0, 1, 0, 0, 0, 0, 2, 0};
    domain.trace.sourceStrengthPrefix = {1.0, 4.0};
    domain.trace.points.resize(12u);
    domain.boundaryTargetDomains.resize(8u, hase::data::invalidDomainId);
    domain.boundaryTargetCells.resize(8u);
    domain.boundaryTargetFaces.resize(8u);
    domain.requestedRays = 7u;

    hase::data::AseDomainGraph graph;
    graph.domains.push_back(std::move(domain));
    graph.globalCellDomains = {3u, 3u};
    auto const costs = hase::core::makeDomainCosts(graph);

    REQUIRE(costs.size() == 1u);
    CHECK(costs.front().id == 3u);
    CHECK(costs.front().cellCount == 2u);
    CHECK(costs.front().boundaryFaceCount == 2u);
    CHECK(costs.front().sourceStrength == Catch::Approx(4.0));
    CHECK(costs.front().requestedRays == 7u);
    CHECK(costs.front().residentBytes > 0u);
}
