#include <backend/primitives/PumpProfile.hpp>

#include <stdexcept>

namespace hase::backend
{
    PumpProfile PumpProfile::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        auto const type = reader.typeName(prefix);
        if(type == "uniformPumpProfile")
            return PumpProfile{UniformPumpProfile::fromTransport(reader, prefix)};
        if(type == "superGaussianPumpProfile")
            return PumpProfile{SuperGaussianPumpProfile::fromTransport(reader, prefix)};
        throw std::runtime_error("unsupported pump profile transport type '" + type + "'");
    }
} // namespace hase::backend
