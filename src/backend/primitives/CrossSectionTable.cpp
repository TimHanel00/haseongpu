#include <backend/primitives/CrossSectionTable.hpp>

namespace hase::backend
{
    CrossSectionTable CrossSectionTable::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        CrossSectionTable result;
        reader.assign(result.wavelengths, prefix, FieldName::wavelengths);
        reader.assign(result.absorption, prefix, FieldName::absorption);
        reader.assign(result.emission, prefix, FieldName::emission);
        reader.assign(result.metadata, prefix, FieldName::metadata);
        return result;
    }
} // namespace hase::backend
