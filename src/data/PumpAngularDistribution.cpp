#include <data/PumpAngularDistribution.hpp>

namespace hase::data
{
    PumpAngularDistribution PumpAngularDistribution::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        PumpAngularDistribution result;
        reader.assign(result.polarAngles, prefix, FieldName::polarAngles);
        reader.assign(result.azimuthalAngles, prefix, FieldName::azimuthalAngles);
        reader.assign(result.weights, prefix, FieldName::weights);
        return result;
    }
} // namespace hase::data
