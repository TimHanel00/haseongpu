#include <backend/primitives/PumpSpectrum.hpp>

namespace hase::backend
{
    PumpSpectrum PumpSpectrum::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        PumpSpectrum result;
        reader.assign(result.wavelengths, prefix, FieldName::wavelengths);
        reader.assign(result.weights, prefix, FieldName::weights);
        return result;
    }
} // namespace hase::backend
