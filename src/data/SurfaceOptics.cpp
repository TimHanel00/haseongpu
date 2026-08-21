#include <data/SurfaceOptics.hpp>

namespace hase::data
{
    SurfaceOptics SurfaceOptics::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        SurfaceOptics result;
        reader.assign(result.reflectivity, prefix, FieldName::reflectivity);
        reader.assign(result.nInside, prefix, FieldName::nInside);
        reader.assign(result.nOutside, prefix, FieldName::nOutside);
        return result;
    }
} // namespace hase::data
