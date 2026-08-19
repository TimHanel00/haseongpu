#pragma once

#include <backend/primitives/Domain.hpp>

#include <memory>
#include <vector>

namespace hase::backend
{
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
} // namespace hase::backend
