#include <data/UniformPumpProfile.hpp>

namespace hase::data
{
    UniformPumpProfile UniformPumpProfile::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        UniformPumpProfile result;
        reader.assign(result.kind, prefix, FieldName::kind);
        return result;
    }
} // namespace hase::data
