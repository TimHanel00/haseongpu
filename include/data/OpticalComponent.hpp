#pragma once

#include <data/Domain.hpp>
#include <data/Material.hpp>
#include <data/SurfaceOpticsAssignment.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hase::data
{
    /**
     * @brief Binds one material and its surface models to a volume domain.
     *
     * An OpticalComponent is the intended unit of a future domain-local trace.
     * Boundary-ray transfer belongs to orchestration between traces; kernels
     * consuming a TraceView never recurse into another component.
     */
    class OpticalComponent
    {
    public:
        struct FieldName
        {
            static constexpr char const* name = "name";
            static constexpr char const* opticalRole = "opticalRole";
            static constexpr char const* domain = "domain";
            static constexpr char const* material = "material";
            static constexpr char const* surfaceOptics = "surfaceOptics";
        };

        std::optional<std::string> name;
        std::optional<std::string> opticalRole;
        std::shared_ptr<Domain> domain;
        std::shared_ptr<Material> material;
        std::vector<std::shared_ptr<SurfaceOpticsAssignment>> surfaceOptics;

        static OpticalComponent fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
