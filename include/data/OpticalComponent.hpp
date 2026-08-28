#pragma once

#include <data/Domain.hpp>
#include <data/Material.hpp>
#include <data/SurfaceOpticsAssignment.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hase::data
{
    /**
     * @brief Binds one material and its surface models to a volume domain.
     *
     * An OpticalComponent is the unit of ASE source allocation and scheduling.
     * Prepared routing metadata connects its boundary faces to adjacent
     * components without adding ownership to the component itself.
     */
    class OpticalComponent
    {
    public:
        struct FieldName
        {
            static constexpr char const* name = "name";
            static constexpr char const* opticalRole = "opticalRole";
            static constexpr char const* aseRays = "aseRays";
            static constexpr char const* domain = "domain";
            static constexpr char const* material = "material";
            static constexpr char const* surfaceOptics = "surfaceOptics";
        };

        std::optional<std::string> name;
        std::optional<std::string> opticalRole;
        std::optional<std::uint64_t> aseRays;
        std::shared_ptr<Domain> domain;
        std::shared_ptr<Material> material;
        std::vector<std::shared_ptr<SurfaceOpticsAssignment>> surfaceOptics;

        /**
         * @brief Read one optical component and its referenced primitives.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the optical-component node.
         * @return Component retaining shared domain, material, and surface-optics references.
         */
        static OpticalComponent fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
