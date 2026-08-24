/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace hase::data
{
    /** @brief Termination state of the forward surface-reservoir iteration. */
    enum class SrmStatus
    {
        disabled,
        converged,
        stable,
        diverged,
        maxIterations
    };

    /**
     * @param status Surface-reservoir termination state.
     * @return Stable lowercase transport spelling of `status`.
     */
    [[nodiscard]] inline char const* toString(SrmStatus const status)
    {
        switch(status)
        {
        case SrmStatus::disabled:
            return "disabled";
        case SrmStatus::converged:
            return "converged";
        case SrmStatus::stable:
            return "stable";
        case SrmStatus::diverged:
            return "diverged";
        case SrmStatus::maxIterations:
            return "max_iterations";
        }
        return "max_iterations";
    }

    /** @brief Cell-ordered ASE estimates, uncertainty, diagnostics, and depletion rates. */
    struct PhiAseResult
    {
        PhiAseResult() = default;

        PhiAseResult(
            std::vector<float> phiAse,
            std::vector<double> standardError,
            std::vector<double> relativeStandardError,
            std::vector<unsigned> totalRays,
            std::vector<double> dndtAse,
            std::vector<unsigned> droppedRays = {},
            SrmStatus srmStatus = SrmStatus::disabled,
            unsigned srmPasses = 0u,
            double srmRemainingFraction = 0.0,
            unsigned srmMaxIterations = 0u,
            unsigned srmDivergenceStreak = 0u)
            : phiAse(std::move(phiAse))
            , standardError(std::move(standardError))
            , relativeStandardError(std::move(relativeStandardError))
            , totalRays(std::move(totalRays))
            , dndtAse(std::move(dndtAse))
            , droppedRays(std::move(droppedRays))
            , srmStatus(srmStatus)
            , srmPasses(srmPasses)
            , srmRemainingFraction(srmRemainingFraction)
            , srmMaxIterations(srmMaxIterations)
            , srmDivergenceStreak(srmDivergenceStreak)
        {
            if(this->droppedRays.empty())
                this->droppedRays.assign(this->phiAse.size(), 0u);
        }

        std::vector<float> phiAse;
        std::vector<double> standardError;
        std::vector<double> relativeStandardError;
        std::vector<unsigned> totalRays;
        std::vector<double> dndtAse;
        std::vector<unsigned> droppedRays;
        SrmStatus srmStatus = SrmStatus::disabled;
        unsigned srmPasses = 0u;
        double srmRemainingFraction = 0.0;
        unsigned srmMaxIterations = 0u;
        unsigned srmDivergenceStreak = 0u;
    };
} // namespace hase::data
