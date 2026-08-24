/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <core/Runtime.hpp>

#include <string_view>

namespace hase::core
{
    /** @brief Effective iteration and divergence limits for reflected-pass tracing. */
    struct SrmControls
    {
        unsigned maxIterations;
        unsigned divergenceStreak;
    };

    /**
     * @param name Environment-variable name.
     * @param fallback Value used when the variable is absent or invalid.
     * @return Parsed positive unsigned value, or `fallback`.
     */
    [[nodiscard]] unsigned positiveEnvironmentUnsigned(std::string_view name, unsigned fallback);

    /**
     * @param experiment User-facing reflection controls.
     * @return Effective SRM limits including supported environment overrides.
     */
    [[nodiscard]] SrmControls resolveSrmControls(AseTraceControls const& experiment);

    /** @return Whether SRM diagnostic logging is enabled by the environment. */
    [[nodiscard]] bool srmDebugLoggingEnabled();

    /**
     * @param status SRM termination state.
     * @return Ordering priority used to merge worker statuses.
     */
    [[nodiscard]] unsigned srmStatusPriority(data::SrmStatus status);

} // namespace hase::core
