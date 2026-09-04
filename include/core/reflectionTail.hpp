/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts/concepts.hpp>
#include <data/PhiAseResult.hpp>
#include <kernels/reflectionTail.hpp>

#include <cstddef>
#include <span>

namespace hase::core
{
    /** @brief Log-linear estimate of the reflected population multiplier. */
    struct BoundaryGammaFit
    {
        double gamma = 0.0;
        double standardError = 0.0;
        std::size_t sampleCount = 0u;
        bool valid = false;
    };

    /** @brief Decision and diagnostics for completing a truncated reflected-pass series. */
    struct BoundaryTailEstimate
    {
        double gamma = 0.0;
        double gammaStandardError = 0.0;
        double tailFactor = 0.0;
        double tailClosure = 0.0;
        bool applicable = false;
        bool divergent = false;
    };

    /**
     * @param residualFractions Positive reflected-weight fractions ordered by pass.
     * @param window Maximum number of trailing samples used by the fit.
     * @return Least-squares fit of log residual weight versus pass number.
     */
    [[nodiscard]] BoundaryGammaFit fitBoundaryGamma(
        std::span<double const> residualFractions,
        std::size_t window = 5u);

    /**
     * @brief Assess whether a truncated reflected-pass series has a stationary finite tail.
     *
     * A Neumann completion is accepted only when the recent multiplier is confidently
     * below one, agrees with a longer-window fit, and the final pass closes the parked
     * reflected weight. A multiplier confidently above one reports divergence.
     *
     * @param residualFractions Reflected weight divided by initial reflected weight.
     * @return Tail factor and classification derived from the pass history.
     */
    [[nodiscard]] BoundaryTailEstimate estimateBoundaryTail(std::span<double const> residualFractions);

    /**
     * @brief Reject failed reflected-ASE fields before coupling them to material evolution.
     *
     * Direct one-state PhiASE calculations may inspect unresolved partial tallies. A time
     * integrator must not consume one: only residual convergence or an accepted analytical
     * tail establishes a finite frozen-inversion field.
     *
     * @param result Reflected-ASE termination diagnostics.
     * @param simulationStep Zero-based material step at which the field was evaluated.
     * @throws std::runtime_error If the reflected field diverged or exhausted its pass limit.
     */
    void requireUsableBoundaryAseForIntegration(data::PhiAseResult const& result, unsigned simulationStep);

    /** Add the accepted Neumann continuation of the final pass to a device accumulator. */
    void applyBoundaryTail(
        concepts::Queue auto const& queue,
        alpaka::concepts::Executor auto const& executor,
        alpaka::concepts::IBuffer<double> auto& accumulated,
        alpaka::concepts::IBuffer<double> auto const& beforeFinalPass,
        double const factor)
    {
        alpaka::onHost::transform(
            queue,
            executor,
            accumulated,
            hase::kernels::CompleteReflectionTail{factor},
            accumulated,
            beforeFinalPass);
    }
} // namespace hase::core
