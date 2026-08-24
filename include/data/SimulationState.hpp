/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <core/Runtime.hpp>
#include <core/SimulationControls.hpp>
#include <data/TraceData.hpp>

namespace hase::data
{
    /** @brief Prepared trace arrays, controls, and result storage for one simulation. */
    struct SimulationState
    {
        core::AseTraceControls ase;
        core::ExecutionPolicy execution;
        TraceData trace;
        PhiAseResult result;
        core::SimulationControls controls;
    };
} // namespace hase::data
