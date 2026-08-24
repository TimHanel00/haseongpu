#pragma once

#include <transport/TransportReader.hpp>

#include <string>

namespace hase::data
{
    /** @brief Uniform spatial pump profile marker. */
    class UniformPumpProfile
    {
    public:
        struct FieldName
        {
            static constexpr char const* kind = "kind";
        };

        std::string kind;

        /**
         * @brief Read a uniform spatial-profile marker.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the profile node.
         * @return Uniform profile retaining its transported kind.
         */
        static UniformPumpProfile fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
