#include <backend/primitives/PlanarPumpRelay.hpp>

namespace hase::backend
{
    PlanarPumpRelay PlanarPumpRelay::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        PlanarPumpRelay result;
        reader.assign(result.flipU, prefix, FieldName::flipU);
        reader.assign(result.flipV, prefix, FieldName::flipV);
        reader.assign(result.rotation, prefix, FieldName::rotation);
        reader.assign(result.offset, prefix, FieldName::offset);
        reader.assign(result.tilt, prefix, FieldName::tilt);
        reader.assign(result.magnification, prefix, FieldName::magnification);
        reader.assign(result.transmission, prefix, FieldName::transmission);
        reader.assign(result.exitDomains, prefix, FieldName::exitDomains);
        reader.assign(result.entryDomains, prefix, FieldName::entryDomains);
        return result;
    }
} // namespace hase::backend
