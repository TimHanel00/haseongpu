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

    /** @brief Scalar residual-fit diagnostics, not evidence of a converged spatial field. */
    struct BoundaryTailEstimate
    {
        double gamma = 0.0;
        double gammaStandardError = 0.0;
        double tailFactor = 0.0;
        double tailClosure = 0.0;
        bool applicable = false; //!< Scalar fit passes its checks; does not authorize field extrapolation.
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
     * The scalar fit is accepted only when the recent multiplier is confidently
     * below one, agrees with a longer-window fit, and the final pass closes the parked
     * reflected weight. This does not establish stationarity of the spatial or spectral
     * distribution and must not be used alone to complete a field. A multiplier
     * confidently above one reports divergence.
     *
     * @param residualFractions Reflected weight divided by initial reflected weight.
     * @return Tail factor and classification derived from the pass history.
     */
    [[nodiscard]] BoundaryTailEstimate estimateBoundaryTail(std::span<double const> residualFractions);

    /**
     * @brief Reject failed reflected-ASE fields before coupling them to material evolution.
     *
     * Direct one-state PhiASE calculations may inspect unresolved partial tallies. A time
     * integrator must not consume one: the current solver requires residual convergence,
     * finite representable output, and no dropped histories.
     *
     * @param result Reflected-ASE termination diagnostics.
     * @param simulationStep Zero-based material step at which the field was evaluated.
     * @throws std::runtime_error If the tally is invalid or boundary propagation is unresolved.
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
