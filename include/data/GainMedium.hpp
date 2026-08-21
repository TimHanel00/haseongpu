#pragma once

#include <data/OpticalComponent.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hase::data
{
    /**
     * @brief A named selection of active optical components.
     *
     * Components retain their own domains and materials. The selection does
     * not merge their topology or spectral data and can therefore later be
     * scheduled as independent domain-local traces.
     */
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
} // namespace hase::data
