#pragma once

#include <backend/primitives/Simulation.hpp>
#include <core/simulationContext.hpp>

#include <cstdint>
#include <vector>

namespace hase::backend::legacy
{
    class ExcitationUpdatePlan
    {
    public:
        ExcitationUpdatePlan(std::vector<std::uint8_t> active, std::vector<std::vector<unsigned>> domainCells);

        [[nodiscard]] std::vector<double> values(ExcitationState const& state) const;
        void apply(ExcitationState const& state, std::vector<double>& destination) const;

    private:
        std::vector<std::uint8_t> m_active;
        std::vector<std::vector<unsigned>> m_domainCells;
    };

    struct LegacyBackendConversion
    {
        core::SimulationContext context;
        ExcitationUpdatePlan excitation;
    };

    class LegacyBackendConverter
    {
    public:
        [[nodiscard]] static core::SimulationContext convert(Simulation const& simulation);
        [[nodiscard]] static LegacyBackendConversion convertWithUpdates(Simulation const& simulation);
    };
} // namespace hase::backend::legacy
