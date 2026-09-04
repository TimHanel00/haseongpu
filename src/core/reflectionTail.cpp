/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <core/reflectionTail.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace hase::core
{
    BoundaryGammaFit fitBoundaryGamma(
        std::span<double const> const residualFractions,
        std::size_t const window)
    {
        if(window < 2u || residualFractions.size() < 2u)
            return {};

        auto const sampleCount = std::min(window, residualFractions.size());
        auto const samples = residualFractions.last(sampleCount);
        if(std::ranges::any_of(samples, [](double const value) { return value <= 0.0 || !std::isfinite(value); }))
            return {};

        double const meanX = 0.5 * static_cast<double>(sampleCount - 1u);
        double meanY = 0.0;
        for(double const value : samples)
            meanY += std::log(value);
        meanY /= static_cast<double>(sampleCount);

        double covariance = 0.0;
        double varianceX = 0.0;
        for(std::size_t sample = 0u; sample < sampleCount; ++sample)
        {
            double const centeredX = static_cast<double>(sample) - meanX;
            covariance += centeredX * (std::log(samples[sample]) - meanY);
            varianceX += centeredX * centeredX;
        }
        if(varianceX <= 0.0)
            return {};

        double const slope = covariance / varianceX;
        double const gamma = std::exp(slope);
        if(!std::isfinite(gamma))
            return {};

        double standardError = std::numeric_limits<double>::infinity();
        if(sampleCount >= 3u)
        {
            double residualSquares = 0.0;
            for(std::size_t sample = 0u; sample < sampleCount; ++sample)
            {
                double const predicted = meanY + slope * (static_cast<double>(sample) - meanX);
                double const residual = std::log(samples[sample]) - predicted;
                residualSquares += residual * residual;
            }
            double const slopeStandardError
                = std::sqrt(residualSquares / static_cast<double>(sampleCount - 2u) / varianceX);
            standardError = gamma * slopeStandardError;
        }
        return BoundaryGammaFit{gamma, standardError, sampleCount, true};
    }

    BoundaryTailEstimate estimateBoundaryTail(std::span<double const> const residualFractions)
    {
        constexpr std::size_t recentWindow = 5u;
        constexpr std::size_t stationarityWindow = 20u;
        constexpr double confidenceMargin = 2.0;
        constexpr double closureTolerance = 0.1;

        BoundaryTailEstimate result;
        auto const recent = fitBoundaryGamma(residualFractions, recentWindow);
        if(!recent.valid)
            return result;
        result.gamma = recent.gamma;
        result.gammaStandardError = recent.standardError;
        if(!std::isfinite(recent.standardError))
            return result;

        result.divergent = recent.gamma - confidenceMargin * recent.standardError > 1.0;
        if(result.divergent || recent.gamma <= 0.0 || recent.gamma >= 1.0
           || recent.gamma + confidenceMargin * recent.standardError >= 1.0
           || residualFractions.size() < stationarityWindow)
            return result;

        auto const longTerm = fitBoundaryGamma(residualFractions, stationarityWindow);
        if(!longTerm.valid || !std::isfinite(longTerm.standardError))
            return result;
        double const combinedStandardError
            = std::hypot(recent.standardError, longTerm.standardError);
        bool const stationary = std::abs(recent.gamma - longTerm.gamma)
                                <= confidenceMargin * combinedStandardError;

        double const previous = residualFractions[residualFractions.size() - 2u];
        double const current = residualFractions.back();
        result.tailFactor = recent.gamma / (1.0 - recent.gamma);
        result.tailClosure = current > 0.0 ? result.tailFactor * (previous - current) / current : 0.0;
        result.applicable = stationary && std::isfinite(result.tailFactor) && result.tailFactor >= 0.0
                            && std::isfinite(result.tailClosure)
                            && std::abs(result.tailClosure - 1.0) <= closureTolerance;
        return result;
    }
} // namespace hase::core
