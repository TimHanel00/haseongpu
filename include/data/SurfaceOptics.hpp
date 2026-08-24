#pragma once

#include <transport/TransportReader.hpp>

namespace hase::data
{
    /** @brief Reflection and refraction properties assigned to a boundary. */
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

        /**
         * @brief Read one boundary optical model from the transport graph.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the surface-optics node.
         * @return Reflectivity and inside/outside refractive indices.
         */
        static SurfaceOptics fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
