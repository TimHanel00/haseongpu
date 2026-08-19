#include <backend/primitives/GainMedium.hpp>

namespace hase::backend
{
    GainMedium GainMedium::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        GainMedium result;
        reader.assign(result.name, prefix, FieldName::name);
        reader.assign(result.components, prefix, FieldName::components);
        return result;
    }
} // namespace hase::backend
