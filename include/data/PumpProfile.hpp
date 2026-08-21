#pragma once

#include <data/SuperGaussianPumpProfile.hpp>
#include <data/UniformPumpProfile.hpp>

#include <variant>

namespace hase::data
{
    /** @brief Closed set of spatial pump profiles accepted by preparation. */
    class PumpProfile
    {
    public:
        std::variant<UniformPumpProfile, SuperGaussianPumpProfile> value;

        static PumpProfile fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
