#include <backend/primitives/ExcitationState.hpp>

namespace hase::backend
{
    ExcitationState ExcitationState::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        ExcitationState result;
        reader.assign(result.values, prefix, FieldName::values);
        reader.assign(result.domains, prefix, FieldName::domains);
        return result;
    }
} // namespace hase::backend
