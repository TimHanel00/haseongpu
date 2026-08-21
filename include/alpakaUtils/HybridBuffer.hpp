

/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts/concepts.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hase::internal::hybridBuffer
{
    template<typename T>
    concept RvalueStorage = !std::is_lvalue_reference_v<T>;

    template<typename T>
    concept HostStorage = !alpaka::onHost::concepts::Device<std::remove_cvref_t<T>>;

    template<typename T_Storage>
    [[nodiscard]] auto makeHostView(T_Storage& storage)
    {
        if constexpr(alpaka::concepts::IView<T_Storage>)
            return alpaka::makeView(storage);
        else
            return alpaka::makeView(
                alpaka::api::host,
                alpaka::onHost::data(storage),
                alpaka::onHost::getExtents(storage));
    }

    template<typename T, std::size_t T_Size>
    [[nodiscard]] auto makeHostView(T (&storage)[T_Size])
    {
        return alpaka::makeView(alpaka::api::host, storage, alpaka::Vec{T_Size});
    }

    template<typename T_HostBuffer, typename T_DeviceBuffer>
    void validateMatchingBuffers(T_HostBuffer const& hostBuffer, T_DeviceBuffer const& deviceBuffer)
    {
        static_assert(
            alpaka::trait::getDim_v<T_HostBuffer> == alpaka::trait::getDim_v<T_DeviceBuffer>,
            "hybrid buffer host and device dimensionality must match");
        static_assert(
            std::same_as<
                std::remove_const_t<alpaka::trait::GetValueType_t<T_HostBuffer>>,
                std::remove_const_t<alpaka::trait::GetValueType_t<T_DeviceBuffer>>>,
            "hybrid buffer host and device value types must match");

        auto const hostExtents = alpaka::onHost::getExtents(hostBuffer);
        auto const deviceExtents = alpaka::onHost::getExtents(deviceBuffer);
        for(std::uint32_t dimension = 0u; dimension < alpaka::trait::getDim_v<T_HostBuffer>; ++dimension)
            if(hostExtents[dimension] != deviceExtents[dimension])
                throw std::invalid_argument("hybrid buffer host and device extents must match");
    }
} // namespace hase::internal::hybridBuffer

namespace hase::alpakaUtils
{
    /**
     * @brief One logical buffer with explicit host and device representations.
     *
     * Construction allocates or associates storage but never transfers data.
     * Lvalue storage is represented by a non-owning Alpaka view; rvalue host
     * storage is owned by the HybridBuffer. Non-owning storage must retain its
     * allocation and outlive the HybridBuffer and all queued operations using
     * its views.
     */
    template<typename T_HostBuffer, typename T_DeviceBuffer>
    class HybridBuffer
    {
    public:
        using HostBuffer = T_HostBuffer;
        using DeviceBuffer = T_DeviceBuffer;

        template<alpaka::onHost::concepts::Device T_Device>
        HybridBuffer(T_Device const& device, T_HostBuffer hostBuffer)
            : m_hostBuffer(std::move(hostBuffer))
            , m_deviceBuffer(alpaka::onHost::allocLike(device, m_hostBuffer))
        {
        }

        HybridBuffer(T_HostBuffer hostBuffer, T_DeviceBuffer deviceBuffer)
            : m_hostBuffer(std::move(hostBuffer))
            , m_deviceBuffer(std::move(deviceBuffer))
        {
            hase::internal::hybridBuffer::validateMatchingBuffers(m_hostBuffer, m_deviceBuffer);
        }

        HybridBuffer(HybridBuffer const&) = delete;
        HybridBuffer& operator=(HybridBuffer const&) = delete;
        HybridBuffer(HybridBuffer&&) = default;
        HybridBuffer& operator=(HybridBuffer&&) = default;

        void toDevice(concepts::Queue auto const& queue)
        {
            alpaka::onHost::memcpy(queue, m_deviceBuffer, m_hostBuffer);
        }

        void toHost(concepts::Queue auto const& queue)
        {
            alpaka::onHost::memcpy(queue, m_hostBuffer, m_deviceBuffer);
            alpaka::onHost::wait(queue);
        }

        [[nodiscard]] auto getHostView()
        {
            return hase::internal::hybridBuffer::makeHostView(m_hostBuffer);
        }

        [[nodiscard]] auto getHostView() const
        {
            return hase::internal::hybridBuffer::makeHostView(m_hostBuffer);
        }

        [[nodiscard]] auto toDeviceView()
        {
            return alpaka::makeView(m_deviceBuffer);
        }

        [[nodiscard]] auto toDeviceView() const
        {
            return alpaka::makeView(m_deviceBuffer);
        }

        [[nodiscard]] auto getExtents() const
        {
            return alpaka::onHost::getExtents(m_hostBuffer);
        }

    private:
        T_HostBuffer m_hostBuffer;
        T_DeviceBuffer m_deviceBuffer;
    };

} // namespace hase::alpakaUtils

namespace hase::internal::hybridBuffer
{
    template<alpaka::onHost::concepts::Device T_Device, typename T_HostBuffer>
    [[nodiscard]] auto makeHybridBuffer(T_Device const& device, T_HostBuffer& hostBuffer)
    {
        using HostBuffer = ALPAKA_TYPEOF(makeHostView(hostBuffer));
        using DeviceBuffer = ALPAKA_TYPEOF(
            alpaka::onHost::allocLike(std::declval<T_Device const&>(), std::declval<HostBuffer const&>()));
        return hase::alpakaUtils::HybridBuffer<HostBuffer, DeviceBuffer>(device, makeHostView(hostBuffer));
    }

    template<alpaka::onHost::concepts::Device T_Device, RvalueStorage T_HostBuffer>
    [[nodiscard]] auto makeHybridBuffer(T_Device const& device, T_HostBuffer&& hostBuffer)
    {
        using HostBuffer = std::remove_cvref_t<T_HostBuffer>;
        using DeviceBuffer = ALPAKA_TYPEOF(
            alpaka::onHost::allocLike(std::declval<T_Device const&>(), std::declval<HostBuffer const&>()));
        return hase::alpakaUtils::HybridBuffer<HostBuffer, DeviceBuffer>(
            device,
            std::forward<T_HostBuffer>(hostBuffer));
    }

    template<HostStorage T_HostBuffer, typename T_DeviceBuffer>
    [[nodiscard]] auto makeHybridBuffer(T_HostBuffer& hostBuffer, T_DeviceBuffer& deviceBuffer)
    {
        using HostBuffer = ALPAKA_TYPEOF(makeHostView(hostBuffer));
        using DeviceBuffer = ALPAKA_TYPEOF(alpaka::makeView(deviceBuffer));
        return hase::alpakaUtils::HybridBuffer<HostBuffer, DeviceBuffer>(
            makeHostView(hostBuffer),
            alpaka::makeView(deviceBuffer));
    }
} // namespace hase::internal::hybridBuffer

namespace hase::alpakaUtils
{

    /** @brief Specializable operation used by getHybridBuffer(). */
    template<typename T_First, typename T_Second>
    struct GetHybridBuffer
    {
        using type = ALPAKA_TYPEOF(
            hase::internal::hybridBuffer::makeHybridBuffer(std::declval<T_First&>(), std::declval<T_Second&>()));

        template<typename T_FirstInput, typename T_SecondInput>
        requires(
            std::same_as<std::remove_cvref_t<T_FirstInput>, T_First>
            && std::same_as<std::remove_cvref_t<T_SecondInput>, T_Second>)
        [[nodiscard]] auto operator()(T_FirstInput&& first, T_SecondInput&& second) const
        {
            return hase::internal::hybridBuffer::makeHybridBuffer(
                std::forward<T_FirstInput>(first),
                std::forward<T_SecondInput>(second));
        }
    };

    template<typename T_First, typename T_Second>
    using GetHybridBuffer_t =
        typename GetHybridBuffer<std::remove_cvref_t<T_First>, std::remove_cvref_t<T_Second>>::type;

    /** @brief Construct a HybridBuffer through the specializable GetHybridBuffer operation. */
    template<typename T_First, typename T_Second>
    [[nodiscard]] auto getHybridBuffer(T_First&& first, T_Second&& second)
    {
        return GetHybridBuffer<std::remove_cvref_t<T_First>, std::remove_cvref_t<T_Second>>{}(
            std::forward<T_First>(first),
            std::forward<T_Second>(second));
    }
} // namespace hase::alpakaUtils
