#pragma once

#include <backend/primitives/Simulation.hpp>
#include <core/simulationContext.hpp>

namespace hase::backend::legacy
{
    class LegacyBackendConverter
    {
    public:
        [[nodiscard]] static core::SimulationContext convert(Simulation const& simulation);
    };
} // namespace hase::backend::legacy
