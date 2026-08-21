#include <data/ExcitationState.hpp>

namespace hase::data
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
} // namespace hase::data
