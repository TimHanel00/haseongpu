#include <backend/primitives/SurfacePumpInjector.hpp>

namespace hase::backend
{
    SurfacePumpInjector SurfacePumpInjector::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        SurfacePumpInjector result;
        reader.assign(result.surfaceDomains, prefix, FieldName::surfaceDomains);
        return result;
    }
} // namespace hase::backend
