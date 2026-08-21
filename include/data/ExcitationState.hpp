#pragma once

#include <data/Domain.hpp>

#include <memory>
#include <vector>

namespace hase::data
{
    /** @brief Domain-indexed excitation values updated between trace iterations. */
    class ExcitationState
    {
    public:
        struct FieldName
        {
            static constexpr char const* values = "values";
            static constexpr char const* domains = "domains";
        };

        transport::RaggedArray<double> values;
        std::vector<std::shared_ptr<Domain>> domains;

        static ExcitationState fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
