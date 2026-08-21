/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <data/TraceData.hpp>
#include <core/SimulationControls.hpp>
#include <core/Runtime.hpp>

namespace hase::data
{
    struct SimulationState
    {
        core::AseTraceControls ase;
        core::ExecutionPolicy execution;
        TraceData trace;
        PhiAseResult result;
        core::SimulationControls controls;
    };
} // namespace hase::data
