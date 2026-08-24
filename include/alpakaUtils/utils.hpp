/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once
#include <alpaka/alpaka.hpp>

#include <utility>

namespace hase::alpakaUtils
{
    template<typename T>
    struct GetApiFromDevice;

    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct GetApiFromDevice<alpaka::onHost::Device<T_Api, T_DeviceKind>>
    {
        using type = T_Api;
    };

    using Vec1D = alpaka::Vec<uint32_t, 1>;
    using Vec2D = alpaka::Vec<uint32_t, 2>;
    using Vec3D = alpaka::Vec<uint32_t, 3>;

    /**
     * @brief Linearize the calling accelerator thread's global index.
     *
     * @param acc Accelerator context supplying grid-thread indices and extents.
     * @return Zero-based linear index within the thread grid.
     */
    ALPAKA_FN_ACC auto getLinGlobalIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        auto idxMd = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        auto extentMd = acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        return alpaka::linearize(extentMd, idxMd);
    }

    template<alpaka::onHost::concepts::Device T_Device>
    using ApiFromDevice = typename hase::alpakaUtils::GetApiFromDevice<T_Device>::type;

    /**
     * @brief Produce an unevaluated API value for a device type.
     *
     * @tparam T_Device Alpaka device whose API type is requested.
     * @return Value of `ApiFromDevice<T_Device>` for compile-time expressions.
     */
    template<alpaka::onHost::concepts::Device T_Device>
    constexpr auto getApiFromDevice()
    {
        return std::declval<ApiFromDevice<T_Device>>();
    }
} // namespace hase::alpakaUtils
