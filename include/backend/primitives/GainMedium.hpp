#pragma once

#include <backend/primitives/OpticalComponent.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hase::backend
{
    class GainMedium
    {
    public:
        struct FieldName
        {
            static constexpr char const* name = "name";
            static constexpr char const* components = "components";
        };

        std::optional<std::string> name;
        std::vector<std::shared_ptr<OpticalComponent>> components;

        static GainMedium fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
