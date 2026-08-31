#include <alpaka/alpaka.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/HybridBuffer.hpp>
#include <alpakaUtils/backendNames.hpp>
#include <alpakaUtils/memory.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/particleComb.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "particle comb selects a fixed population and keeps the candidate total on device",
    "[forward][boundary][backend]",
    TestBackends)
{
    auto const backend = TestType::makeDict();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(backend);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << alpaka::onHost::DeviceSpec{backend}.getName());
        return;
    }
    auto device = deviceSelector.makeDevice(0);
    auto const executor = alpaka::getExecutor(backend);
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    hase::alpakaUtils::DevBundle devBundle(device, executor);
    auto weights = hase::alpakaUtils::toDevice(queue, std::vector<double>{1.0, 3.0, 6.0});
    hase::core::ParticleCombWorkspace workspace(device, 6u, 4u);

    workspace.enqueue(devBundle, queue, weights.getView(), 3u, 4u, 1234u, 17u);
    std::vector<std::uint32_t> selected(4u);
    auto selectedTransfer = hase::alpakaUtils::getHybridBuffer(selected, workspace.selectedView(4u));
    selectedTransfer.toHost(queue);
    std::vector<double> total(1u);
    auto totalTransfer = hase::alpakaUtils::getHybridBuffer(total, workspace.totalWeightView());
    totalTransfer.toHost(queue);
    std::vector<double> selectedWeights(4u);
    auto selectedWeightTransfer
        = hase::alpakaUtils::getHybridBuffer(selectedWeights, workspace.selectedWeightsView(4u));
    selectedWeightTransfer.toHost(queue);

    CHECK(total.front() == Catch::Approx(10.0));
    CHECK(std::ranges::all_of(selected, [](auto const index) { return index < 3u; }));
    CHECK(std::ranges::is_sorted(selected));
    CHECK(total.front() / static_cast<double>(selected.size()) == Catch::Approx(2.5));
    CHECK(std::ranges::all_of(selectedWeights, [](double const weight) { return weight == Catch::Approx(2.5); }));
}

TEMPLATE_LIST_TEST_CASE(
    "particle comb preserves requested populations and weight per domain",
    "[forward][boundary][domain][backend]",
    TestBackends)
{
    auto const backend = TestType::makeDict();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(backend);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << alpaka::onHost::DeviceSpec{backend}.getName());
        return;
    }
    auto device = deviceSelector.makeDevice(0);
    auto const executor = alpaka::getExecutor(backend);
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    hase::alpakaUtils::DevBundle devBundle(device, executor);
    auto weights = hase::alpakaUtils::toDevice(queue, std::vector<double>{1.0, 3.0, 6.0});
    std::vector<std::uint32_t> const candidateDomains{0u, 0u, 1u};
    auto domains = hase::alpakaUtils::toDevice(queue, candidateDomains);
    hase::core::ParticleCombWorkspace workspace(device, 3u, 4u);

    workspace.enqueueDomain(devBundle, queue, weights.getView(), domains.getView(), 3u, 0u, 0u, 2u, 41u, 0u);
    workspace.enqueueDomain(devBundle, queue, weights.getView(), domains.getView(), 3u, 1u, 2u, 2u, 41u, 1u);
    workspace.enqueueTotal(devBundle, queue, weights.getView(), 3u);

    std::vector<std::uint32_t> selected(4u);
    auto selectedTransfer = hase::alpakaUtils::getHybridBuffer(selected, workspace.selectedView(4u));
    selectedTransfer.toHost(queue);
    std::vector<double> selectedWeights(4u);
    auto weightTransfer = hase::alpakaUtils::getHybridBuffer(selectedWeights, workspace.selectedWeightsView(4u));
    weightTransfer.toHost(queue);
    REQUIRE(selected.size() == 4u);
    CHECK(candidateDomains[selected[0u]] == 0u);
    CHECK(candidateDomains[selected[1u]] == 0u);
    CHECK(candidateDomains[selected[2u]] == 1u);
    CHECK(candidateDomains[selected[3u]] == 1u);
    CHECK(selectedWeights[0u] == Catch::Approx(2.0));
    CHECK(selectedWeights[1u] == Catch::Approx(2.0));
    CHECK(selectedWeights[2u] == Catch::Approx(3.0));
    CHECK(selectedWeights[3u] == Catch::Approx(3.0));
}

TEMPLATE_LIST_TEST_CASE(
    "spatial particle comb retains route weight without duplicating boundary candidates",
    "[forward][boundary][domain][spatial][backend]",
    TestBackends)
{
    auto const backend = TestType::makeDict();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(backend);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << alpaka::onHost::DeviceSpec{backend}.getName());
        return;
    }
    auto device = deviceSelector.makeDevice(0);
    auto const executor = alpaka::getExecutor(backend);
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    hase::alpakaUtils::DevBundle devBundle(device, executor);
    auto weightBuffer = hase::alpakaUtils::toDevice(queue, std::vector<double>{0.4, 0.4, 0.5, 0.5, 0.3, 0.3});
    auto domainBuffer = hase::alpakaUtils::toDevice(queue, std::vector<std::uint32_t>{0u, 1u, 0u, 1u, 0u, 1u});
    auto cellBuffer = hase::alpakaUtils::toDevice(queue, std::vector<std::uint32_t>{0u, 10u, 1u, 11u, 2u, 12u});
    auto faceBuffer = hase::alpakaUtils::toDevice(queue, std::vector<std::uint32_t>(6u, 0u));
    auto barycentricX = hase::alpakaUtils::toDevice(queue, std::vector<double>{0.1, 0.1, 0.5, 0.5, 0.9, 0.9});
    auto barycentricY = hase::alpakaUtils::toDevice(queue, std::vector<double>(6u, 0.0));
    auto positionZ = hase::alpakaUtils::toDevice(queue, std::vector<double>(6u, 0.0));

    struct Barycentric
    {
        ALPAKA_TYPEOF(barycentricX.getView()) x;
        ALPAKA_TYPEOF(barycentricY.getView()) y;
    };

    struct Position
    {
        ALPAKA_TYPEOF(barycentricX.getView()) x;
        ALPAKA_TYPEOF(barycentricY.getView()) y;
        ALPAKA_TYPEOF(positionZ.getView()) z;
    };

    struct Candidates
    {
        ALPAKA_TYPEOF(weightBuffer.getView()) weights;
        ALPAKA_TYPEOF(domainBuffer.getView()) targetDomains;
        ALPAKA_TYPEOF(cellBuffer.getView()) targetCells;
        ALPAKA_TYPEOF(faceBuffer.getView()) targetFaces;
        Position positions;
        Barycentric faceBarycentric;
    };

    auto const candidates = Candidates{
        weightBuffer.getView(),
        domainBuffer.getView(),
        cellBuffer.getView(),
        faceBuffer.getView(),
        {barycentricX.getView(), barycentricY.getView(), positionZ.getView()},
        {barycentricX.getView(), barycentricY.getView()}};
    hase::core::ParticleCombWorkspace workspace(device, 6u, 3u, 2u);

    workspace.enqueueSpatialDomain(devBundle, queue, candidates, 6u, 3u, 0u, 0u, 2u, 5489u, 0u);
    workspace.enqueueSpatialDomain(devBundle, queue, candidates, 6u, 3u, 1u, 2u, 1u, 5489u, 1u);
    std::vector<std::uint32_t> selected(3u);
    auto selectedTransfer = hase::alpakaUtils::getHybridBuffer(selected, workspace.selectedView(3u));
    selectedTransfer.toHost(queue);
    std::vector<double> selectedWeights(3u);
    auto weightTransfer = hase::alpakaUtils::getHybridBuffer(selectedWeights, workspace.selectedWeightsView(3u));
    weightTransfer.toHost(queue);

    CHECK(selected[0u] != selected[1u]);
    CHECK(selected[0u] % 2u == 0u);
    CHECK(selected[1u] % 2u == 0u);
    CHECK(selected[2u] % 2u == 1u);
    CHECK(selectedWeights[0u] + selectedWeights[1u] == Catch::Approx(1.2));
    CHECK(selectedWeights[2u] == Catch::Approx(1.2));
}
