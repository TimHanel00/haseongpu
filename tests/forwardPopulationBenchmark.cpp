#include "include/populationCube.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/calcForwardPhiAse.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>

namespace
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

    struct EnsembleMoments
    {
        std::uint32_t count{};
        double mean{}, squareSum{};

        void add(double value)
        {
            auto const delta = value - mean;
            mean += delta / ++count;
            squareSum += delta * (value - mean);
        }

        double variance() const
        {
            return squareSum / (count - 1u);
        }
    };
} // namespace

// Core-to-core comparison: both paths share resident inputs, raw downloads and
// host finalization. Setup is excluded; source preparation is timed in both paths.
TEMPLATE_LIST_TEST_CASE("compare logical and legacy SRM batches", "[.][srmPopulationBenchmark]", TestBackends)
{
    auto selector = alpaka::onHost::makeDeviceSelector(TestType::makeDict());
    if(!selector.isAvailable())
        SKIP("Requested test backend has no available device");
    auto device = selector.makeDevice(0u);
    auto const executor = alpaka::getExecutor(TestType::makeDict());
    using Context = hase::core::ForwardPhiAseDeviceContext<ALPAKA_TYPEOF(device), ALPAKA_TYPEOF(executor)>;
    constexpr std::uint32_t ensembleSize = 64u;
    std::cout << std::setprecision(12);
    for(auto const subdivisions : {2u, 6u})
    {
        auto mesh = hase::test::populationCube(subdivisions, 0.25f);
        auto graph = hase::test::populationSources(mesh);
        graph.globalCellDomains.assign(mesh.numberOfCells, 0u);
        auto resident = mesh.makeResident(device);
        hase::core::ResidentAseDomainSources sources(device, graph);
        hase::core::ResidentAseDomainInterfaces interfaces(device, graph);
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        resident.toDevice(queue);
        sources.toDevice(queue);
        interfaces.toDevice(queue);
        alpaka::onHost::wait(queue);
        hase::core::AseTraceControls controls;
        controls.reflectionMode = "srm";
        controls.useReflections = true;
        controls.forwardRayCount = controls.minRays = controls.maxRays = 8192u;
        controls.numIndependentRayPopulations = 8u;
        controls.boundaryMaxPasses = 40u;
        controls.reflectionTolerance = 1.0e-4;
        controls.enableDiagnostics = false;
        std::array<std::unique_ptr<Context>, 2u> contexts{
            std::make_unique<Context>(device, executor, controls, mesh),
            std::make_unique<Context>(device, executor, controls, mesh)};
        auto const total = mesh.sourceStrengthPrefix.back();
        std::vector<hase::core::DomainQuota> quotas{{0u, controls.forwardRayCount, total, 0.0}};
        std::array counts{controls.forwardRayCount};
        auto const plan = hase::core::makeForwardPopulationBatches(
            counts,
            quotas,
            total,
            controls.numIndependentRayPopulations,
            hase::core::maxLogicalSrmBatchRays);
        std::array<std::vector<EnsembleMoments>, 2u> moments;
        std::array<std::vector<double>, 2u> reported, timings;
        std::vector<EnsembleMoments> differences(mesh.numberOfCells);
        for(std::uint32_t mode = 0u; mode < 2u; ++mode)
        {
            moments[mode].resize(mesh.numberOfCells);
            reported[mode].resize(mesh.numberOfCells, 0.0);
        }
        for(std::uint32_t run = 0u; run <= ensembleSize; ++run)
        {
            auto const seed = hase::random::seedForAdaptiveLaunch(136u + run, 0u);
            std::array<hase::data::PhiAseResult, 2u> results;
            for(std::uint32_t order = 0u; order < 2u; ++order)
            {
                auto const mode = (order + run) % 2u;
                auto& context = *contexts[mode];
                auto const start = std::chrono::steady_clock::now();
                if(mode == 0u)
                    for(std::uint32_t population = 0u; population < controls.numIndependentRayPopulations;
                        ++population)
                    {
                        std::array const domainCounts{hase::core::domainPopulationRayCount(
                            controls.forwardRayCount,
                            population,
                            controls.numIndependentRayPopulations)};
                        std::array const weights{1.0};
                        context.begin(
                            resident.view(),
                            domainCounts[0u],
                            seed,
                            population,
                            total,
                            controls,
                            interfaces.view(),
                            sources.view(),
                            domainCounts,
                            weights,
                            domainCounts,
                            population == 0u);
                    }
                else
                {
                    context.prepareRayPopulations(resident.view(), sources.view(), plan, seed);
                    context.traceLogicalSrmBatches(resident.view(), interfaces.view(), controls, seed);
                }
                hase::core::ForwardPhiAseRawResult raw;
                float ignoredRuntime{};
                context.finish(raw, ignoredRuntime);
                hase::core::finalizeForwardPhiAse(mesh, raw, total, results[mode]);
                auto const elapsed
                    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                REQUIRE(raw.rayCount == controls.forwardRayCount);
                REQUIRE(raw.boundaryStatus == hase::data::BoundaryStatus::converged);
                if(run == 0u)
                    continue;
                timings[mode].push_back(elapsed);
                for(std::uint32_t cell = 0u; cell < mesh.numberOfCells; ++cell)
                {
                    REQUIRE(std::isfinite(results[mode].standardError[cell]));
                    moments[mode][cell].add(results[mode].phiAse[cell]);
                    reported[mode][cell] += std::pow(results[mode].standardError[cell], 2) / ensembleSize;
                }
            }
            if(run != 0u)
                for(std::uint32_t cell = 0u; cell < mesh.numberOfCells; ++cell)
                    differences[cell].add(static_cast<double>(results[1u].phiAse[cell]) - results[0u].phiAse[cell]);
        }
        double differenceSquare = 0.0, differenceNoise = 0.0;
        for(std::uint32_t cell = 0u; cell < mesh.numberOfCells; ++cell)
        {
            differenceSquare += mesh.cellVolumes[cell] * std::pow(differences[cell].mean, 2);
            differenceNoise += mesh.cellVolumes[cell] * differences[cell].variance() / ensembleSize;
        }
        for(std::uint32_t mode = 0u; mode < 2u; ++mode)
        {
            double observed = 0.0, estimated = 0.0;
            for(std::uint32_t cell = 0u; cell < mesh.numberOfCells; ++cell)
            {
                observed += mesh.cellVolumes[cell] * moments[mode][cell].variance();
                estimated += mesh.cellVolumes[cell] * reported[mode][cell];
            }
            std::ranges::sort(timings[mode]);
            std::cout << "SRM_SUMMARY,cells=" << mesh.numberOfCells << ",rays=" << controls.forwardRayCount
                      << ",seeds=" << ensembleSize << ",mode=" << (mode == 0u ? "legacy" : "logical")
                      << ",medianMs=" << (timings[mode][31u] + timings[mode][32u]) / 2.0
                      << ",observedVariance=" << observed << ",reportedVariance=" << estimated
                      << ",meanDifferenceRms=" << std::sqrt(differenceSquare)
                      << ",meanDifferenceNoiseRms=" << std::sqrt(differenceNoise) << std::endl;
        }
    }
}
