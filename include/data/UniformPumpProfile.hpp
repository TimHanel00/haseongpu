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

        static UniformPumpProfile fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
