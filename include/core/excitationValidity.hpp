#pragma once

#include <alpakaUtils/HybridBuffer.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>

namespace hase::internal::timeIntegration
{
    struct NonFiniteValue
    {
        ALPAKA_FN_ACC std::uint32_t operator()(double const value) const
        {
            return alpaka::math::isfinite(value) ? 0u : 1u;
        }
    };
} // namespace hase::internal::timeIntegration

namespace hase::core
{
    /** Device-side validity reduction; only its scalar result crosses to the host. */
    template<alpaka::onHost::concepts::Device T_Device>
    class ExcitationValidity
    {
        std::array<std::uint32_t, 1u> m_invalidCountHost{};
        alpakaUtils::GetHybridBuffer_t<T_Device, std::array<std::uint32_t, 1u>> m_invalidCount;

    public:
        explicit ExcitationValidity(T_Device const& device)
            : m_invalidCount(alpakaUtils::getHybridBuffer(device, m_invalidCountHost))
        {
        }

        /** Reject non-finite stage values before they can be consumed or clipped. */
        void requireFinite(
            concepts::Queue auto const& queue,
            alpaka::concepts::Executor auto const executor,
            alpaka::concepts::IView<double> auto const beta)
        {
            alpaka::onHost::transformReduce(
                queue,
                executor,
                std::uint32_t{0u},
                m_invalidCount.toDeviceView(),
                std::plus{},
                alpaka::ScalarFunc{internal::timeIntegration::NonFiniteValue{}},
                beta);
            m_invalidCount.toHost(queue);
            if(m_invalidCount.getHostView()[0u] != 0u)
                throw std::runtime_error("Non-finite excitation in material integration; the step was not accepted.");
        }
    };
} // namespace hase::core
