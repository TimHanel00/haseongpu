#include <data/SurfaceOpticsAssignment.hpp>

namespace hase::data
{
    SurfaceOpticsAssignment SurfaceOpticsAssignment::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        SurfaceOpticsAssignment result;
        reader.assign(result.domain, prefix, FieldName::domain);
        reader.assign(result.optics, prefix, FieldName::optics);
        return result;
    }
} // namespace hase::data
