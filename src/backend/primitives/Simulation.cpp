#include <backend/primitives/Simulation.hpp>

namespace hase::backend
{
    Simulation Simulation::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        Simulation result;
        reader.assign(result.timeStep, prefix, FieldName::timeStep);
        reader.assign(result.simulationSteps, prefix, FieldName::simulationSteps);
        reader.assign(result.endTime, prefix, FieldName::endTime);
        reader.assign(result.prePump, prefix, FieldName::prePump);
        reader.assign(result.executionMode, prefix, FieldName::executionMode);
        reader.assign(result.outputSteps, prefix, FieldName::outputSteps);
        reader.assign(result.outputFields, prefix, FieldName::outputFields);
        reader.assign(result.controlFields, prefix, FieldName::controlFields);
        reader.assign(result.currentStep, prefix, FieldName::currentStep);
        reader.assign(result.currentTime, prefix, FieldName::currentTime);
        reader.assign(result.opticalComponents, prefix, FieldName::opticalComponents);
        reader.assign(result.gainMedium, prefix, FieldName::gainMedium);
        reader.assign(result.exteriorSurface, prefix, FieldName::exteriorSurface);
        reader.assign(result.excitationState, prefix, FieldName::excitationState);
        reader.assign(result.phiAse, prefix, FieldName::phiAse);
        reader.assign(result.timeIntegrationSolver, prefix, FieldName::timeIntegrationSolver);
        reader.assign(result.pumpRegistrations, prefix, FieldName::pumpRegistrations);
        return result;
    }
} // namespace hase::backend
