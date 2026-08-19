#include <backend/primitives/TimeIntegrationSolver.hpp>

namespace hase::backend
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
} // namespace hase::backend
