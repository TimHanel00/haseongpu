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

        /**
         * @brief Read domain-indexed excitation values from one graph node.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the excitation-state node.
         * @return Excitation values with identity-preserving domain references.
         */
        static ExcitationState fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
