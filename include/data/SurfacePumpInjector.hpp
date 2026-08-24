#pragma once

#include <data/Domain.hpp>

#include <memory>
#include <vector>

namespace hase::data
{
    /** @brief Selects boundary domains from which one pump spawns rays. */
    class SurfacePumpInjector
    {
    public:
        struct FieldName
        {
            static constexpr char const* surfaceDomains = "surfaceDomains";
        };

        std::vector<std::shared_ptr<Domain>> surfaceDomains;

        /**
         * @brief Read the boundary domains used to inject one pump.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the injector node.
         * @return Injector retaining identity-preserving surface-domain references.
         */
        static SurfacePumpInjector fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
