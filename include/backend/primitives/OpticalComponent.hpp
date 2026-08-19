#pragma once

#include <backend/primitives/Domain.hpp>
#include <backend/primitives/Material.hpp>
#include <backend/primitives/SurfaceOpticsAssignment.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hase::backend
{
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
} // namespace hase::backend
