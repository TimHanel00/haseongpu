#pragma once

#include <data/Domain.hpp>
#include <data/SurfaceOptics.hpp>

#include <memory>

namespace hase::data
{
    /** @brief Binds surface optics to one domain boundary selection. */
    class SurfaceOpticsAssignment
    {
    public:
        struct FieldName
        {
            static constexpr char const* domain = "domain";
            static constexpr char const* optics = "optics";
        };

        std::shared_ptr<Domain> domain;
        std::shared_ptr<SurfaceOptics> optics;

        /**
         * @brief Read a surface-domain to optics association.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the assignment node.
         * @return Assignment retaining shared domain and optics references.
         */
        static SurfaceOpticsAssignment fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
