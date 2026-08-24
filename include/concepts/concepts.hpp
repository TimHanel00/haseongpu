/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once
#include <alpaka/alpaka.hpp>

#include <concepts>
#include <type_traits>

namespace hase::concepts
{
    template<typename T>
    struct IsQueue : std::false_type
    {
    };

    template<
        alpaka::concepts::Api T_Api,
        alpaka::concepts::DeviceKind T_DeviceKind,
        alpaka::concepts::QueuePolicyList T_QueuePolicies>
    struct IsQueue<alpaka::onHost::Queue<alpaka::onHost::Device<T_Api, T_DeviceKind>, T_QueuePolicies>>
        : std::true_type
    {
    };

    /** @brief Type models an Alpaka host queue after cv-reference removal. */
    template<typename T>
    concept Queue = IsQueue<std::remove_cvref_t<T>>::value;

    /**
     * @brief Logical buffer exposing host and device views of `T_Data`.
     *
     * @tparam T Candidate logical-buffer type.
     * @tparam T_Data Required element type of both views.
     *
     * Matching extents are available through `getExtents()`. The concept does
     * not imply which representation is current; transfers remain explicit.
     */
    template<typename T, typename T_Data>
    concept HybridBuffer = requires(T& buffer) {
        requires alpaka::concepts::IView<decltype(buffer.getHostView()), T_Data>;
        requires alpaka::concepts::IView<decltype(buffer.toDeviceView()), T_Data>;
        buffer.getExtents();
    };
} // namespace hase::concepts
