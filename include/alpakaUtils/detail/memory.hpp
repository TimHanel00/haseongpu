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

#include <vector>

namespace hase::alpakaUtils::detail
{
    /**
     * @brief Default host-container-to-device allocation and synchronous copy operation.
     * @tparam T Host container or Alpaka view accepted by `allocLike` and `memcpy`.
     *
     * Invoked through the public `hase::alpakaUtils::toDevice` customization point.
     */
    template<typename T>
    struct ToDevice
    {
        /**
         * @param queue Queue whose device owns the returned allocation.
         * @param inputView Contiguous host-accessible source used to infer type and extent.
         * @return Owning device buffer populated with `inputView`; the copy is complete on return.
         */
        auto operator()(concepts::Queue auto const& queue, T const& inputView)
        {
            auto deviceBuffer = alpaka::onHost::allocLike(queue.getDevice(), inputView);
            alpaka::onHost::memcpy(queue, deviceBuffer, inputView);
            alpaka::onHost::wait(queue);
            return deviceBuffer;
        }
    };

    /**
     * @brief Alpaka-view specialization retaining an explicit first template argument.
     *
     * A fully variadic `alpaka::View<TArgs...>` specialization triggers an nvcc
     * 12.9 compiler crash. Naming the first argument preserves the same dispatch
     * while avoiding that compiler defect.
     */
    template<typename A, typename... TArgs>
    struct ToDevice<alpaka::View<A, TArgs...>>
    {
        /**
         * @param queue Queue whose device owns the returned allocation.
         * @param inputView Host-accessible Alpaka view defining the element type and extent.
         * @return Owning device buffer populated from the view; the copy is complete on return.
         */
        auto operator()(concepts::Queue auto const& queue, alpaka::View<A, TArgs...> const& inputView)
        {
            auto deviceBuffer = alpaka::onHost::allocLike(queue.getDevice(), inputView);
            alpaka::onHost::memcpy(queue, deviceBuffer, inputView);
            alpaka::onHost::wait(queue);
            return deviceBuffer;
        }
    };

    /** @brief `std::vector` specialization used because it does not model the Alpaka view interface. */
    template<typename A>
    struct ToDevice<std::vector<A>>
    {
        /**
         * @param queue Queue whose device owns the returned allocation.
         * @param inputBuffer Contiguous vector copied in full.
         * @return Owning device buffer populated from the vector; the copy is complete on return.
         */
        auto operator()(concepts::Queue auto const& queue, std::vector<A> const& inputBuffer)
        {
            auto deviceBuffer = alpaka::onHost::allocLike(queue.getDevice(), inputBuffer);
            alpaka::onHost::memcpy(queue, deviceBuffer, inputBuffer);
            alpaka::onHost::wait(queue);
            return deviceBuffer;
        }
    };
} // namespace hase::alpakaUtils::detail
