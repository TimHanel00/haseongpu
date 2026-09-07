#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/HybridBuffer.hpp>
#include <alpakaUtils/memory.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/calcForwardPhiAse.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

    struct RecordEssentialFailure
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto scores,
            auto visits,
            auto dropped) const
        {
            using namespace hase::kernels::forward;
            auto const accumulation
                = ForwardAccumulationSpans{scores.getMdSpan(), visits.getMdSpan(), dropped.getMdSpan()};
            auto cellPolicy = MakeForwardAseCellPolicy{}(tracePolicy::diagnostics::none, accumulation);
            cellPolicy.cellDiagnostics.recordVisit(acc, 0u);
            cellPolicy.cellDiagnostics.recordDropped(acc, 0u);
            auto failure = MakeForwardRayFailureBehaviour{}(tracePolicy::diagnostics::none, accumulation);
            ForwardAseRayState ray;
            ray.cell = 0u;
            failure(acc, hase::data::TraceView{}, ray);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "host and device ASE finalization agree for unequal batches and invalid outputs",
    "[forward][finalization][backend]",
    TestBackends)
{
    auto const backend = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(backend);
    if(!selector.isAvailable())
    {
        SKIP("Requested test backend has no available device");
    }
    auto device = selector.makeDevice(0u);
    auto const executor = alpaka::getExecutor(backend);
    auto queue = device.makeQueue(alpaka::queueKind::blocking);
    hase::alpakaUtils::DevBundle devBundle(device, executor);
    hase::data::TraceData mesh;
    mesh.numberOfCells = 1u;
    mesh.numberOfMeshPoints = 4u;
    mesh.numberOfMaterials = 1u;
    mesh.numberOfCellVertices = 4u;
    mesh.cellPointIndices = {0u, 1u, 2u, 3u};
    mesh.cellVolumes = {1.0f};
    mesh.cellMaterialIds = {0u};
    mesh.materialActive = {1u};
    mesh.betaVolume = {1.0};
    mesh.materialPeakEmission = {1.0};
    mesh.materialPeakAbsorption = {0.0};
    auto points = hase::alpakaUtils::toDevice(queue, mesh.cellPointIndices);
    auto volumes = hase::alpakaUtils::toDevice(queue, mesh.cellVolumes);
    auto materialIds = hase::alpakaUtils::toDevice(queue, mesh.cellMaterialIds);
    auto active = hase::alpakaUtils::toDevice(queue, mesh.materialActive);
    auto beta = hase::alpakaUtils::toDevice(queue, mesh.betaVolume);
    auto emission = hase::alpakaUtils::toDevice(queue, mesh.materialPeakEmission);
    auto absorption = hase::alpakaUtils::toDevice(queue, mesh.materialPeakAbsorption);
    auto lumpedVolumes = hase::alpakaUtils::toDevice(queue, std::vector<double>(4u, 0.25));
    hase::data::TraceView view{};
    view.numberOfCells = 1u;
    view.numberOfMeshPoints = 4u;
    view.numberOfMaterials = 1u;
    view.numberOfCellVertices = 4u;
    view.cellPointIndices = {points.data(), 4u};
    view.cellVolumes = {volumes.data(), 1u};
    view.cellMaterialIds = {materialIds.data(), 1u};
    view.materialActive = {active.data(), 1u};
    view.betaVolume = {beta.data(), 1u};
    view.materialPeakEmission = {emission.data(), 1u};
    view.materialPeakAbsorption = {absorption.data(), 1u};
    view.lumpedMaterialVertexVolumes = {lumpedVolumes.data(), 4u};

    for(unsigned scenario = 0u; scenario < 4u; ++scenario)
    {
        CAPTURE(scenario);
        auto raw = hase::core::makeForwardRawResult(1u, 4u, 2u);
        raw.rayCount = 3u;
        raw.rseBatchRayCounts = {2u, 1u};
        raw.vertexBatchScoreSum = {0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75};
        if(scenario == 2u)
            raw.droppedRays[0u] = 1u;
        if(scenario == 3u)
            raw.vertexBatchScoreSum[0u] = std::numeric_limits<double>::infinity();
        double const sourceStrength = scenario == 1u ? 1.0e39 : 1.0;
        hase::data::PhiAseResult hostResult;
        hase::core::finalizeForwardPhiAse(mesh, raw, sourceStrength, hostResult);
        auto scores = hase::alpakaUtils::toDevice(queue, raw.vertexBatchScoreSum);
        auto counts = hase::alpakaUtils::toDevice(queue, raw.rseBatchRayCounts);
        auto dropped = hase::alpakaUtils::toDevice(queue, raw.droppedRays);
        auto phi = hase::alpakaUtils::getHybridBuffer(device, std::array<float, 1u>{});
        auto error = hase::alpakaUtils::getHybridBuffer(device, std::array<double, 1u>{});
        auto rse = hase::alpakaUtils::getHybridBuffer(device, std::array<double, 1u>{});
        auto rate = hase::alpakaUtils::getHybridBuffer(device, std::array<double, 1u>{});
        auto const frame = alpaka::onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u}, executor};
        queue.enqueue(
            frame,
            alpaka::KernelBundle{
                hase::kernels::FinalizeForwardVolumePhiAse{3u, 2u, sourceStrength},
                view,
                std::as_const(scores).getView(),
                counts.getView(),
                std::as_const(dropped).getView(),
                phi.toDeviceView(),
                error.toDeviceView(),
                rse.toDeviceView(),
                rate.toDeviceView()});
        phi.toHost(queue);
        error.toHost(queue);
        rse.toHost(queue);
        rate.toHost(queue);
        if(scenario == 1u || scenario == 3u)
        {
            CHECK_FALSE(std::isfinite(phi.getHostView()[0u]));
            CHECK_FALSE(std::isfinite(hostResult.phiAse[0u]));
        }
        else
        {
            CHECK(phi.getHostView()[0u] == Catch::Approx(2.0));
            CHECK(phi.getHostView()[0u] == Catch::Approx(hostResult.phiAse[0u]));
            CHECK(rate.getHostView()[0u] == Catch::Approx(hostResult.dndtAse[0u]));
        }
        CHECK(rse.getHostView()[0u] == Catch::Approx(hostResult.relativeStandardError[0u]));
        CHECK(error.getHostView()[0u] == Catch::Approx(hostResult.standardError[0u]));
    }
}

TEMPLATE_LIST_TEST_CASE(
    "failure accounting remains enabled without visit diagnostics",
    "[forward][validity][backend]",
    TestBackends)
{
    auto const backend = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(backend);
    if(!selector.isAvailable())
        SKIP("Requested test backend has no available device");
    auto device = selector.makeDevice(0u);
    auto const executor = alpaka::getExecutor(backend);
    auto queue = device.makeQueue(alpaka::queueKind::blocking);
    auto scores = hase::alpakaUtils::getHybridBuffer(device, std::array<double, 1u>{});
    auto visits = hase::alpakaUtils::getHybridBuffer(device, std::array<std::uint32_t, 1u>{});
    auto dropped = hase::alpakaUtils::getHybridBuffer(device, std::array<std::uint32_t, 1u>{});
    scores.toDevice(queue);
    visits.toDevice(queue);
    dropped.toDevice(queue);
    auto const frame = alpaka::onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u}, executor};
    queue.enqueue(
        frame,
        alpaka::KernelBundle{
            RecordEssentialFailure{},
            scores.toDeviceView(),
            visits.toDeviceView(),
            dropped.toDeviceView()});
    visits.toHost(queue);
    dropped.toHost(queue);
    CHECK(visits.getHostView()[0u] == 0u);
    CHECK(dropped.getHostView()[0u] == 2u);
}
