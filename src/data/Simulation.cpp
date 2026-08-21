#include <data/Simulation.hpp>

#include <stdexcept>
#include <unordered_set>

namespace hase::data
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

    void Simulation::updateFromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        if(!reader.dynamicOnly())
            throw std::runtime_error("a full transport iteration cannot be applied as a dynamic update");
        if(!excitationState)
            throw std::runtime_error("simulation has no excitation state to update");
        reader.prefetch(prefix);
        reader.assign(currentStep, prefix, FieldName::currentStep);
        reader.assign(currentTime, prefix, FieldName::currentTime);
        auto const paths = reader.referencePaths(prefix.child(FieldName::excitationState).string());
        if(paths.size() != 1u)
            throw std::runtime_error("dynamic simulation update requires one excitation-state reference");
        auto const statePath = transport::TransportPath{paths.front()};
        reader.assign(excitationState->values, statePath, ExcitationState::FieldName::values);

        std::unordered_set<CrossSectionTable*> updatedTables;
        for(auto const& component : opticalComponents)
        {
            if(!component || !component->material || !component->material->crossSections)
                continue;
            auto& table = *component->material->crossSections;
            if(updatedTables.insert(&table).second
               && reader.contains(table.transportPath, CrossSectionTable::FieldName::wavelengths))
                table.updateFromTransport(reader, table.transportPath);
        }
    }
} // namespace hase::data
