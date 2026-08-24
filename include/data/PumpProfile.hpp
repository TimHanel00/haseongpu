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

        /**
         * @brief Read and dispatch one supported spatial pump profile.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the profile node, including its transported type.
         * @return Closed profile variant matching the node type.
         */
        static PumpProfile fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
