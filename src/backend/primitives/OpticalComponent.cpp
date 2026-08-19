#include <backend/primitives/OpticalComponent.hpp>

namespace hase::backend
{
    OpticalComponent OpticalComponent::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        OpticalComponent result;
        reader.assign(result.name, prefix, FieldName::name);
        reader.assign(result.opticalRole, prefix, FieldName::opticalRole);
        reader.assign(result.domain, prefix, FieldName::domain);
        reader.assign(result.material, prefix, FieldName::material);
        reader.assign(result.surfaceOptics, prefix, FieldName::surfaceOptics);
        return result;
    }
} // namespace hase::backend
