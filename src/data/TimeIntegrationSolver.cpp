#include <data/TimeIntegrationSolver.hpp>

namespace hase::data
{
    TimeIntegrationSolver TimeIntegrationSolver::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        TimeIntegrationSolver result;
        reader.assign(result.name, prefix, FieldName::name);
        reader.assign(result.iterations, prefix, FieldName::iterations);
        reader.assign(result.tolerance, prefix, FieldName::tolerance);
        return result;
    }
} // namespace hase::data
