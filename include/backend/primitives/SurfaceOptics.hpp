#pragma once

#include <backend/transport/TransportReader.hpp>

namespace hase::backend
{
    class SurfaceOptics
    {
    public:
        struct FieldName
        {
            static constexpr char const* reflectivity = "reflectivity";
            static constexpr char const* nInside = "nInside";
            static constexpr char const* nOutside = "nOutside";
        };

        double reflectivity{};
        double nInside{1.0};
        double nOutside{1.0};

        static SurfaceOptics fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
