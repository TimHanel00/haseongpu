#pragma once

#include <backend/primitives/Domain.hpp>

#include <memory>
#include <vector>

namespace hase::backend
{
    class SurfacePumpInjector
    {
    public:
        struct FieldName
        {
            static constexpr char const* surfaceDomains = "surfaceDomains";
        };

        std::vector<std::shared_ptr<Domain>> surfaceDomains;

        static SurfacePumpInjector fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
