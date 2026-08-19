#pragma once

#include <backend/transport/TransportReader.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace hase::backend
{
    class TimeIntegrationSolver
    {
    public:
        struct FieldName
        {
            static constexpr char const* name = "name";
            static constexpr char const* iterations = "iterations";
            static constexpr char const* tolerance = "tolerance";
        };

        std::string name;
        std::optional<std::uint64_t> iterations;
        std::optional<double> tolerance;

        static TimeIntegrationSolver fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
