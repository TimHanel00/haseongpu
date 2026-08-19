#include <backend/primitives/PumpRegistration.hpp>

namespace hase::backend
{
    PumpRegistration PumpRegistration::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        PumpRegistration result;
        reader.assign(result.pump, prefix, FieldName::pump);
        reader.assign(result.injectionMethod, prefix, FieldName::injectionMethod);
        reader.assign(result.relays, prefix, FieldName::relays);
        return result;
    }
} // namespace hase::backend
