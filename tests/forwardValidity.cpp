#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/reflectionTail.hpp>

#include <limits>

TEST_CASE("material integration rejects invalid ASE even without reflections", "[forward][correctness]")
{
    hase::data::PhiAseResult result;
    result.phiAse = {1.0f};
    result.dndtAse = {1.0};
    result.droppedRays = {0u};
    CHECK_NOTHROW(hase::core::requireUsableBoundaryAseForIntegration(result, 0u));
    result.droppedRays[0u] = 1u;
    CHECK_THROWS_AS(hase::core::requireUsableBoundaryAseForIntegration(result, 0u), std::runtime_error);
    result.droppedRays[0u] = 0u;
    result.phiAse[0u] = std::numeric_limits<float>::infinity();
    CHECK_THROWS_AS(hase::core::requireUsableBoundaryAseForIntegration(result, 0u), std::runtime_error);
    result.phiAse[0u] = 1.0f;
    result.dndtAse[0u] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(hase::core::requireUsableBoundaryAseForIntegration(result, 0u), std::runtime_error);
}
