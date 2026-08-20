#pragma once

#include <backend/primitives/Domain.hpp>
#include <backend/primitives/ExcitationState.hpp>
#include <backend/primitives/GainMedium.hpp>
#include <backend/primitives/OpticalComponent.hpp>
#include <backend/primitives/PhiAse.hpp>
#include <backend/primitives/PumpRegistration.hpp>
#include <backend/primitives/TimeIntegrationSolver.hpp>
#include <backend/transport/TransportReader.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hase::backend
{
    class Simulation
    {
    public:
        struct FieldName
        {
            static constexpr char const* timeStep = "timeStep";
            static constexpr char const* simulationSteps = "simulationSteps";
            static constexpr char const* endTime = "endTime";
            static constexpr char const* prePump = "prePump";
            static constexpr char const* executionMode = "executionMode";
            static constexpr char const* outputSteps = "outputSteps";
            static constexpr char const* outputFields = "outputFields";
            static constexpr char const* controlFields = "controlFields";
            static constexpr char const* currentStep = "currentStep";
            static constexpr char const* currentTime = "currentTime";
            static constexpr char const* opticalComponents = "opticalComponents";
            static constexpr char const* gainMedium = "gainMedium";
            static constexpr char const* exteriorSurface = "exteriorSurface";
            static constexpr char const* excitationState = "excitationState";
            static constexpr char const* phiAse = "phiAse";
            static constexpr char const* timeIntegrationSolver = "timeIntegrationSolver";
            static constexpr char const* pumpRegistrations = "pumpRegistrations";
        };

        double timeStep{};
        std::optional<std::uint64_t> simulationSteps;
        std::optional<double> endTime;
        bool prePump{};
        std::string executionMode;
        std::optional<transport::Array<std::uint64_t>> outputSteps;
        std::vector<std::string> outputFields;
        std::vector<std::string> controlFields;
        std::uint64_t currentStep{};
        double currentTime{};
        std::vector<std::shared_ptr<OpticalComponent>> opticalComponents;
        std::shared_ptr<GainMedium> gainMedium;
        std::shared_ptr<Domain> exteriorSurface;
        std::shared_ptr<ExcitationState> excitationState;
        std::shared_ptr<PhiAse> phiAse;
        std::shared_ptr<TimeIntegrationSolver> timeIntegrationSolver;
        std::vector<std::shared_ptr<PumpRegistration>> pumpRegistrations;

        static Simulation fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);

        void updateFromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix);
    };
} // namespace hase::backend
