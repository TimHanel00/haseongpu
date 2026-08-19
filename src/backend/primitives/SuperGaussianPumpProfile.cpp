#include <backend/primitives/SuperGaussianPumpProfile.hpp>

namespace hase::backend
{
    SuperGaussianPumpProfile SuperGaussianPumpProfile::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        SuperGaussianPumpProfile result;
        reader.assign(result.kind, prefix, FieldName::kind);
        reader.assign(result.radiusU, prefix, FieldName::radiusU);
        reader.assign(result.radiusV, prefix, FieldName::radiusV);
        reader.assign(result.exponent, prefix, FieldName::exponent);
        reader.assign(result.center, prefix, FieldName::center);
        reader.assign(result.axisU, prefix, FieldName::axisU);
        reader.assign(result.axisV, prefix, FieldName::axisV);
        return result;
    }
} // namespace hase::backend
