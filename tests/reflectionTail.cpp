/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <core/reflectionTail.hpp>

#include <cmath>
#include <vector>

TEST_CASE("boundary gamma fit recovers a geometric reflected population", "[forward][reflection-tail]")
{
    std::vector<double> residualFractions;
    for(unsigned pass = 0u; pass < 24u; ++pass)
        residualFractions.push_back(std::pow(0.9, static_cast<double>(pass)));

    auto const fit = hase::core::fitBoundaryGamma(residualFractions);

    REQUIRE(fit.valid);
    CHECK(fit.gamma == Catch::Approx(0.9).epsilon(1.0e-12));
    CHECK(fit.standardError == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("stationary subcritical reflection tail is completed geometrically", "[forward][reflection-tail]")
{
    std::vector<double> residualFractions;
    for(unsigned pass = 0u; pass < 24u; ++pass)
        residualFractions.push_back(std::pow(0.9, static_cast<double>(pass)));

    auto const estimate = hase::core::estimateBoundaryTail(residualFractions);

    CHECK_FALSE(estimate.divergent);
    REQUIRE(estimate.applicable);
    CHECK(estimate.tailFactor == Catch::Approx(9.0).epsilon(1.0e-12));
    CHECK(estimate.tailClosure == Catch::Approx(1.0).epsilon(1.0e-12));
}

TEST_CASE("confidently growing reflected population is divergent", "[forward][reflection-tail]")
{
    std::vector<double> residualFractions;
    for(unsigned pass = 0u; pass < 24u; ++pass)
        residualFractions.push_back(std::pow(1.02, static_cast<double>(pass)));

    auto const estimate = hase::core::estimateBoundaryTail(residualFractions);

    CHECK(estimate.divergent);
    CHECK_FALSE(estimate.applicable);
    CHECK(estimate.tailFactor == 0.0);
}

TEST_CASE("spectrally hardening reflected population refuses a stationary tail", "[forward][reflection-tail]")
{
    std::vector<double> residualFractions{1.0};
    for(unsigned pass = 1u; pass < 30u; ++pass)
    {
        double const gamma = 0.8 + 0.006 * static_cast<double>(pass);
        residualFractions.push_back(residualFractions.back() * gamma);
    }

    auto const estimate = hase::core::estimateBoundaryTail(residualFractions);

    CHECK_FALSE(estimate.divergent);
    CHECK_FALSE(estimate.applicable);
}
