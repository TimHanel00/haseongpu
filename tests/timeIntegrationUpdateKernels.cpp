#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <kernels/timeIntegrationUpdateKernels.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

TEST_CASE("frozen-source derivative changes only with stage decay", "[time-integration][simd]")
{
    std::vector<std::uint8_t> const materialActive{1u, 0u};
    std::vector<double> const fluorescenceLifetimes{2.0, 0.0};
    hase::kernels::ComposeFrozenSourcesDerivative const compose{materialActive, fluorescenceLifetimes};

    alpaka::Simd<double, 4u> const betaVolume{0.2, 0.7, 0.4, 0.9};
    alpaka::Simd<double, 4u> const dndtPump{0.3, 0.8, 0.5, 1.0};
    alpaka::Simd<double, 4u> const dndtAse{0.1, 0.2, 0.4, 0.1};
    alpaka::Simd<unsigned, 4u> const materialIds{0u, 1u, 0u, 1u};

    auto const result = compose(betaVolume, dndtPump, dndtAse, materialIds);
    CHECK(result[0u] == Catch::Approx(0.3 - 0.1 - 0.2 / 2.0));
    CHECK(result[1u] == 0.0);
    CHECK(result[2u] == Catch::Approx(0.5 - 0.4 - 0.4 / 2.0));
    CHECK(result[3u] == 0.0);
}

TEST_CASE("exponential Euler update supports mixed-material SIMD packs", "[time-integration][simd]")
{
    std::vector<std::uint8_t> const materialActive{1u, 0u};
    std::vector<double> const fluorescenceLifetimes{2.0, 0.0};
    hase::kernels::ExponentialEulerUpdate const update{0.5, materialActive, fluorescenceLifetimes};

    alpaka::Simd<double, 4u> const betaVolume{0.2, 0.7, 0.4, 0.9};
    alpaka::Simd<double, 4u> const dndtPump{0.3, 0.8, 0.5, 1.0};
    alpaka::Simd<double, 4u> const dndtAse{0.1, 0.2, 0.4, 0.1};
    alpaka::Simd<unsigned, 4u> const materialIds{0u, 1u, 0u, 1u};

    auto const result = update(betaVolume, dndtPump, dndtAse, materialIds);
    double const decay = std::exp(-0.5 / 2.0);
    CHECK(result[0u] == Catch::Approx(2.0 * (0.3 - 0.1) * (1.0 - decay) + 0.2 * decay));
    CHECK(result[1u] == 0.0);
    CHECK(result[2u] == Catch::Approx(2.0 * (0.5 - 0.4) * (1.0 - decay) + 0.4 * decay));
    CHECK(result[3u] == 0.0);
}
