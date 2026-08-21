#include <alpaka/alpaka.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/HybridBuffer.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hase::tests
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors))>;

    struct Increment
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, auto values) const
        {
            for(auto const index : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{values.getExtents()}))
                values[index] += 1;
        }
    };

    struct WrappedHostStorage
    {
        std::array<int, 2u> values;
    };
} // namespace hase::tests

namespace hase::alpakaUtils
{
    template<alpaka::onHost::concepts::Device T_Device>
    struct GetHybridBuffer<T_Device, hase::tests::WrappedHostStorage>
    {
        using type = GetHybridBuffer_t<T_Device, std::array<int, 2u>>;

        [[nodiscard]] auto operator()(T_Device& device, hase::tests::WrappedHostStorage& hostStorage) const -> type
        {
            return getHybridBuffer(device, hostStorage.values);
        }
    };
} // namespace hase::alpakaUtils

TEMPLATE_LIST_TEST_CASE(
    "hybrid buffer transfers explicit host and device views",
    "[alpaka][memory]",
    hase::tests::TestBackends)
{
    auto const backend = TestType::makeDict();
    auto deviceSelector = alpaka::onHost::makeDeviceSelector(backend[alpaka::object::deviceSpec]);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << backend[alpaka::object::deviceSpec].getName());
        return;
    }
    auto device = deviceSelector.makeDevice(0);
    auto const executor = backend[alpaka::object::exec];
    auto queue = device.makeQueue(alpaka::queueKind::blocking);

    SECTION("owns moved host storage")
    {
        auto buffer = hase::alpakaUtils::getHybridBuffer(device, std::vector<int>{1, 2, 3, 4});
        STATIC_REQUIRE_FALSE(std::copy_constructible<decltype(buffer)>);
        REQUIRE(buffer.getExtents()[0u] == 4u);

        buffer.toDevice(queue);
        auto const frameSpec = hase::alpakaUtils::getFrameSpec<unsigned>(device, executor, buffer.getExtents());
        queue.enqueue(frameSpec, alpaka::KernelBundle{hase::tests::Increment{}, buffer.toDeviceView()});
        buffer.toHost(queue);

        auto hostView = buffer.getHostView();
        CHECK(hostView[0u] == 2);
        CHECK(hostView[1u] == 3);
        CHECK(hostView[2u] == 4);
        CHECK(hostView[3u] == 5);
    }

    SECTION("references writable stack storage")
    {
        int values[]{2, 4, 6};
        auto buffer = hase::alpakaUtils::getHybridBuffer(device, values);
        buffer.getHostView()[1u] = 5;
        buffer.toDevice(queue);

        auto const frameSpec = hase::alpakaUtils::getFrameSpec<unsigned>(device, executor, buffer.getExtents());
        queue.enqueue(frameSpec, alpaka::KernelBundle{hase::tests::Increment{}, buffer.toDeviceView()});
        buffer.toHost(queue);

        CHECK(values[0u] == 3);
        CHECK(values[1u] == 6);
        CHECK(values[2u] == 7);
    }

    SECTION("preserves multidimensional Alpaka extents")
    {
        auto hostBuffer = alpaka::onHost::allocHost<int>(alpaka::Vec{2u, 3u});
        auto buffer = hase::alpakaUtils::getHybridBuffer(device, hostBuffer);
        REQUIRE(buffer.getExtents()[0u] == 2u);
        REQUIRE(buffer.getExtents()[1u] == 3u);

        auto hostView = buffer.getHostView();
        for(auto const index : alpaka::IdxRange{hostView.getExtents()})
            hostView[index] = static_cast<int>(index.y() * 3u + index.x());
        buffer.toDevice(queue);

        auto const frameSpec = hase::alpakaUtils::getFrameSpec<unsigned>(device, executor, buffer.getExtents());
        queue.enqueue(frameSpec, alpaka::KernelBundle{hase::tests::Increment{}, buffer.toDeviceView()});
        buffer.toHost(queue);

        CHECK(buffer.getHostView()[alpaka::Vec{1u, 2u}] == 6);
    }

    SECTION("associates existing host and device storage")
    {
        std::array<int, 2u> hostStorage{7, 9};
        auto deviceStorage = alpaka::onHost::allocLike(device, hostStorage);
        auto buffer = hase::alpakaUtils::getHybridBuffer(hostStorage, deviceStorage);

        buffer.toDevice(queue);
        auto const frameSpec = hase::alpakaUtils::getFrameSpec<unsigned>(device, executor, buffer.getExtents());
        queue.enqueue(frameSpec, alpaka::KernelBundle{hase::tests::Increment{}, buffer.toDeviceView()});
        buffer.toHost(queue);

        CHECK(hostStorage == std::array<int, 2u>{8, 10});
    }

    SECTION("rejects mismatched existing storage")
    {
        std::array<int, 2u> hostStorage{};
        auto deviceStorage = alpaka::onHost::alloc<int>(device, 3u);
        CHECK_THROWS_AS(hase::alpakaUtils::getHybridBuffer(hostStorage, deviceStorage), std::invalid_argument);
    }

    SECTION("routes construction through specializations")
    {
        hase::tests::WrappedHostStorage hostStorage{{11, 13}};
        using ExpectedBuffer = hase::alpakaUtils::GetHybridBuffer_t<decltype(device), hase::tests::WrappedHostStorage>;
        auto buffer = hase::alpakaUtils::getHybridBuffer(device, hostStorage);
        STATIC_REQUIRE(std::same_as<decltype(buffer), ExpectedBuffer>);

        buffer.toDevice(queue);
        auto const frameSpec = hase::alpakaUtils::getFrameSpec<unsigned>(device, executor, buffer.getExtents());
        queue.enqueue(frameSpec, alpaka::KernelBundle{hase::tests::Increment{}, buffer.toDeviceView()});
        buffer.toHost(queue);

        CHECK(hostStorage.values == std::array<int, 2u>{12, 14});
    }
}
