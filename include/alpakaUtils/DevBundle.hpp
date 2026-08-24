/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once
#include <alpaka/alpaka.hpp>

namespace hase::alpakaUtils
{
    /**
     * @brief Select a backend-appropriate frame specification.
     *
     * @tparam T_DataType Element type used by backends whose frame selection
     * depends on the data type.
     * @param device Device on which the frame will execute.
     * @param executor Executor used when Alpaka provides an executor-aware overload.
     * @param extents Logical work-item extents.
     * @return Alpaka frame specification for the requested work extent.
     */
    template<typename T_DataType>
    auto getFrameSpec(
        alpaka::onHost::concepts::Device auto const& device,
        alpaka::concepts::Executor auto const& executor,
        alpaka::concepts::Vector auto const& extents)
    {
        if constexpr(requires { alpaka::onHost::getFrameSpec(device, executor, extents); })
        {
            return alpaka::onHost::getFrameSpec(device, executor, extents);
        }
        else
        {
            return alpaka::onHost::getFrameSpec<T_DataType>(device, extents);
        }
    }

    /**
     * @brief Device and executor pair passed to HASE enqueue interfaces.
     *
     * @tparam T_Device Alpaka host device type.
     * @tparam T_Executor Executor used to construct frame specifications.
     */
    template<alpaka::onHost::concepts::Device T_Device, alpaka::concepts::Executor T_Executor>
    struct DevBundle
    {
        T_Device device;
        T_Executor executor;

        DevBundle(T_Device const& device, T_Executor const& executor) : device(device), executor(executor)
        {
        }
    };
} // namespace hase::alpakaUtils
