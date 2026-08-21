#include <data/Domain.hpp>

namespace hase::data
{
    Domain Domain::fromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        Domain result;
        reader.assign(result.entityKind, prefix, FieldName::entityKind);
        reader.assign(result.masks, prefix, FieldName::masks);
        reader.assign(result.topologies, prefix, FieldName::topologies);
        return result;
    }
} // namespace hase::data
