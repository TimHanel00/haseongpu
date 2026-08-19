#pragma once

#include <backend/primitives/SuperGaussianPumpProfile.hpp>
#include <backend/primitives/UniformPumpProfile.hpp>

#include <variant>

namespace hase::backend
{
    class PumpProfile
    {
    public:
        std::variant<UniformPumpProfile, SuperGaussianPumpProfile> value;

        static PumpProfile fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
