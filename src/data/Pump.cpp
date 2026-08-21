#include <data/Pump.hpp>

namespace hase::data
{
    Pump Pump::fromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        Pump result;
        reader.assign(result.totalPower, prefix, FieldName::totalPower);
        reader.assign(result.rayCount, prefix, FieldName::rayCount);
        reader.assign(result.pumpSteps, prefix, FieldName::pumpSteps);
        reader.assign(result.rngSeed, prefix, FieldName::rngSeed);
        reader.assign(result.name, prefix, FieldName::name);
        reader.assign(result.spectrum, prefix, FieldName::spectrum);
        reader.assign(result.profile, prefix, FieldName::profile);
        reader.assign(result.angularDistribution, prefix, FieldName::angularDistribution);
        return result;
    }
} // namespace hase::data
