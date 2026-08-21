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

        static SurfaceOpticsAssignment fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
