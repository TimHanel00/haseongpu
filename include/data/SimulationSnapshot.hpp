/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <data/PhiAseResult.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace hase::data
{
    /** @brief Selected fields published for one completed simulation step. */
    struct SimulationSnapshot
    {
        unsigned step = 0u;
        double time = 0.0;
        std::vector<double> betaVolume;
        PhiAseResult aseResult;
        std::vector<double> dndtPump;
        std::vector<double> dndtAse;
        std::vector<std::string> fields;

        /**
         * @param field Stable simulation-output field name.
         * @return Whether this snapshot includes that field.
         */
        [[nodiscard]] bool contains(std::string const& field) const
        {
            return std::ranges::find(fields, field) != fields.end();
        }
    };
} // namespace hase::data
