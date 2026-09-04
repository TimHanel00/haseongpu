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
    /** @brief Termination state of domain-boundary propagation. */
    enum class BoundaryStatus
    {
        disabled,
        converged,
        stable,
        diverged,
        maxPasses
    };

    /** @brief Analytical completion state of a truncated reflected-pass series. */
    enum class BoundaryTailStatus
    {
        none,
        applied,
        refused
    };

    /**
     * @param status Domain-boundary termination state.
     * @return Stable lowercase transport spelling of `status`.
     */
    [[nodiscard]] inline char const* toString(BoundaryStatus const status)
    {
        switch(status)
        {
        case BoundaryStatus::disabled:
            return "disabled";
        case BoundaryStatus::converged:
            return "converged";
        case BoundaryStatus::stable:
            return "stable";
        case BoundaryStatus::diverged:
            return "diverged";
        case BoundaryStatus::maxPasses:
            return "maxPasses";
        }
        return "maxPasses";
    }

    /** @return Stable lowercase transport spelling of `status`. */
    [[nodiscard]] inline char const* toString(BoundaryTailStatus const status)
    {
        switch(status)
        {
        case BoundaryTailStatus::none:
            return "none";
        case BoundaryTailStatus::applied:
            return "applied";
        case BoundaryTailStatus::refused:
            return "refused";
        }
        return "refused";
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
            BoundaryStatus boundaryStatus = BoundaryStatus::disabled,
            unsigned boundaryPasses = 0u,
            double boundaryRemainingFraction = 0.0,
            unsigned boundaryMaxPasses = 0u,
            unsigned boundaryDivergenceStreak = 0u,
            BoundaryTailStatus boundaryTailStatus = BoundaryTailStatus::none,
            double boundaryGamma = 0.0,
            double boundaryGammaStandardError = 0.0,
            double boundaryTailFactor = 0.0,
            double boundaryTailClosure = 0.0)
            : phiAse(std::move(phiAse))
            , standardError(std::move(standardError))
            , relativeStandardError(std::move(relativeStandardError))
            , totalRays(std::move(totalRays))
            , dndtAse(std::move(dndtAse))
            , droppedRays(std::move(droppedRays))
            , boundaryStatus(boundaryStatus)
            , boundaryPasses(boundaryPasses)
            , boundaryRemainingFraction(boundaryRemainingFraction)
            , boundaryMaxPasses(boundaryMaxPasses)
            , boundaryDivergenceStreak(boundaryDivergenceStreak)
            , boundaryTailStatus(boundaryTailStatus)
            , boundaryGamma(boundaryGamma)
            , boundaryGammaStandardError(boundaryGammaStandardError)
            , boundaryTailFactor(boundaryTailFactor)
            , boundaryTailClosure(boundaryTailClosure)
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
        BoundaryStatus boundaryStatus = BoundaryStatus::disabled;
        unsigned boundaryPasses = 0u;
        double boundaryRemainingFraction = 0.0;
        unsigned boundaryMaxPasses = 0u;
        unsigned boundaryDivergenceStreak = 0u;
        BoundaryTailStatus boundaryTailStatus = BoundaryTailStatus::none;
        double boundaryGamma = 0.0;
        double boundaryGammaStandardError = 0.0;
        double boundaryTailFactor = 0.0;
        double boundaryTailClosure = 0.0;
    };
} // namespace hase::data
