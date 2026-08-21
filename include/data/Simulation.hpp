#pragma once

#include <data/Domain.hpp>
#include <data/ExcitationState.hpp>
#include <data/GainMedium.hpp>
#include <data/OpticalComponent.hpp>
#include <data/PhiAse.hpp>
#include <data/PumpRegistration.hpp>
#include <data/TimeIntegrationSolver.hpp>
#include <transport/TransportReader.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hase::data
{
    /**
     * @brief Root of the transported physical and numerical primitive graph.
     *
     * The graph remains authoritative on the host. Execution preparation
     * derives domain-local arrays without creating a second semantic object
     * model, and explicit synchronized updates mutate this same graph.
     */
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

        /**
         * @brief Apply fields explicitly marked dynamic in one transport step.
         *
         * Cross-section updates replace complete material tables. Topology and
         * component-to-material assignment remain immutable during a run.
         */
        void updateFromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix);
    };
} // namespace hase::data
