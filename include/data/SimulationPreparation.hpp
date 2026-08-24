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
        /**
         * @brief Store the prepared mapping from excitation domains to cells.
         * @param active Per-cell active-material mask in prepared cell order.
         * @param domainCells Cell indices selected by each excitation domain.
         */
        ExcitationLayout(std::vector<std::uint8_t> active, std::vector<std::vector<unsigned>> domainCells);

        /**
         * @param state Domain-indexed excitation values to project.
         * @return One excitation value per prepared cell; inactive cells are zero.
         */
        [[nodiscard]] std::vector<double> values(ExcitationState const& state) const;

        /**
         * @brief Project excitation values into existing prepared cell storage.
         * @param state Domain-indexed excitation values to project.
         * @param destination Cell-ordered output vector to replace.
         */
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

    /**
     * @brief Prepare the primitive graph for a direct ASE trace.
     * @param simulation Host-authoritative primitive graph.
     * @return Flattened execution state with prepared tracing arrays and controls.
     */
    [[nodiscard]] SimulationState prepareSimulation(Simulation const& simulation);

    /**
     * @brief Prepare a time-stepped run and retain its excitation projection.
     * @param simulation Host-authoritative primitive graph.
     * @return Prepared execution state plus the reusable excitation layout.
     */
    [[nodiscard]] SimulationPreparation prepareSimulationWithUpdates(Simulation const& simulation);
} // namespace hase::data
