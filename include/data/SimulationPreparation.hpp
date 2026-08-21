#pragma once

#include <data/Simulation.hpp>
#include <data/SimulationState.hpp>

#include <cstdint>
#include <vector>

namespace hase::data
{
    /** @brief Maps domain-shaped excitation values onto the prepared cell order. */
    class ExcitationLayout
    {
    public:
        ExcitationLayout(std::vector<std::uint8_t> active, std::vector<std::vector<unsigned>> domainCells);

        [[nodiscard]] std::vector<double> values(ExcitationState const& state) const;
        void apply(ExcitationState const& state, std::vector<double>& destination) const;

    private:
        std::vector<std::uint8_t> m_active;
        std::vector<std::vector<unsigned>> m_domainCells;
    };

    /**
     * @brief Device-oriented state derived once from the primitive graph.
     *
     * This is preparation, not a second public object model: semantic values
     * remain owned by the primitives, while TraceData contains the flattened
     * arrays required by local ray kernels.
     */
    struct SimulationPreparation
    {
        SimulationState state;
        ExcitationLayout excitation;
    };

    /** @brief Prepare the primitive graph for a direct ASE trace. */
    [[nodiscard]] SimulationState prepareSimulation(Simulation const& simulation);

    /** @brief Prepare a time-stepped run and retain its excitation projection. */
    [[nodiscard]] SimulationPreparation prepareSimulationWithUpdates(Simulation const& simulation);
} // namespace hase::data
