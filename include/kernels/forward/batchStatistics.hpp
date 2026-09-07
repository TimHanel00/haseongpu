#pragma once

#include <alpaka/alpaka.hpp>

#include <limits>

namespace hase::kernels::forward
{
    /** Independent, complete-source replicate means; history counts may differ. */
    struct BatchStatistics
    {
        unsigned count = 0u;
        double mean = 0.0;
        double centeredSquareSum = 0.0;

        ALPAKA_FN_HOST_ACC void add(double const scoreDensity, unsigned const histories)
        {
            if(histories == 0u)
                return;
            double const value = scoreDensity / static_cast<double>(histories);
            double const delta = value - mean;
            mean += delta / static_cast<double>(++count);
            centeredSquareSum += delta * (value - mean);
        }

        struct Estimate
        {
            double value;
            double standardError = std::numeric_limits<double>::max();
            double relativeStandardError = std::numeric_limits<double>::max();
        };

        ALPAKA_FN_HOST_ACC Estimate finalize(double const sourceStrength, bool const dropped) const
        {
            Estimate result{mean * sourceStrength};
            if(!alpaka::math::isfinite(result.value)
               || alpaka::math::abs(result.value) > static_cast<double>(std::numeric_limits<float>::max())
               || !alpaka::math::isfinite(centeredSquareSum))
            {
                result.value = std::numeric_limits<double>::quiet_NaN();
                return result;
            }
            if(dropped || count < 2u)
                return result;
            result.standardError
                = alpaka::math::sqrt(
                      alpaka::math::max(0.0, centeredSquareSum) / (static_cast<double>(count) * (count - 1u)))
                  * alpaka::math::abs(sourceStrength);
            result.relativeStandardError = result.value == 0.0
                                               ? std::numeric_limits<double>::quiet_NaN()
                                               : result.standardError / alpaka::math::abs(result.value);
            return result;
        }
    };
} // namespace hase::kernels::forward
