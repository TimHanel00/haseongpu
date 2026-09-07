#include <alpakaUtils/memory.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/excitationValidity.hpp>
#include <kernels/timeIntegrationUpdateKernels.hpp>

#include <array>
#include <limits>
#include <type_traits>
#include <vector>

using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE(
    "excitation validation rejects non-finite updates before clipping",
    "[integration]",
    TestBackends)
{
    auto const backend = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(backend);
    if(!selector.isAvailable())
        SKIP("Requested test backend has no available device");
    auto device = selector.makeDevice(0u);
    auto const executor = alpaka::getExecutor(backend);
    auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
    hase::core::ExcitationValidity validator(device);
    for(double const invalid :
        {std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()})
    {
        auto beta = hase::alpakaUtils::toDevice(queue, std::vector<double>{0.5, invalid, 0.2});
        CHECK_THROWS_AS(validator.requireFinite(queue, executor, beta.getView()), std::runtime_error);
    }
    auto beta = hase::alpakaUtils::getHybridBuffer(device, std::array<double, 3u>{-0.5, 0.5, 1.5});
    beta.toDevice(queue);
    CHECK_NOTHROW(validator.requireFinite(queue, executor, beta.toDeviceView()));
    alpaka::onHost::transform(queue, executor, beta.toDeviceView(), hase::kernels::ClipBeta{}, beta.toDeviceView());
    beta.toHost(queue);
    CHECK(beta.getHostView()[0u] == 0.0);
    CHECK(beta.getHostView()[1u] == 0.5);
    CHECK(beta.getHostView()[2u] == 1.0);

    auto huge = hase::alpakaUtils::toDevice(queue, std::vector<double>{std::numeric_limits<double>::max()});
    auto out = alpaka::onHost::alloc<double>(device, std::size_t{1u});
    alpaka::onHost::transform(queue, executor, out, hase::kernels::AddScaled{2.0}, huge, huge);
    CHECK_THROWS_AS(validator.requireFinite(queue, executor, out.getView()), std::runtime_error);
}
