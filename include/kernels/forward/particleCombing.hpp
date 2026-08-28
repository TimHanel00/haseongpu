#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <span>

namespace hase::kernels::forward
{
    /** @brief Select one domain's candidate weights without moving candidate records. */
    struct FilterParticleCombDomainWeights
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IView<double> auto const candidateWeights,
            alpaka::concepts::IView<std::uint32_t> auto const candidateDomains,
            alpaka::concepts::IView<double> auto filteredWeights,
            std::uint32_t const candidateCount,
            std::uint32_t const domain) const
        {
            for(auto [candidate] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{candidateCount}))
                filteredWeights[candidate] = candidateDomains[candidate] == domain ? candidateWeights[candidate] : 0.0;
        }
    };

    /** @brief Select face weights whose owning cells belong to one domain. */
    struct FilterParticleCombFaceDomainWeights
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IView<double> auto const faceWeights,
            std::span<std::uint32_t const> const cellDomains,
            alpaka::concepts::IView<double> auto filteredWeights,
            std::uint32_t const faceCount,
            std::uint32_t const facesPerCell,
            std::uint32_t const domain) const
        {
            for(auto [face] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{faceCount}))
                filteredWeights[face] = cellDomains[face / facesPerCell] == domain ? faceWeights[face] : 0.0;
        }
    };

    struct CaptureParticleCombTotalWeight
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IView<double> auto const cdf,
            alpaka::concepts::IView<double> auto totalWeight,
            std::uint32_t const candidateCount) const
        {
            for(auto [index] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{1u}))
                totalWeight[index] = candidateCount == 0u ? 0.0 : cdf[candidateCount - 1u];
        }
    };

    struct GenerateParticleCombOffset
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IView<double> auto const totalWeight,
            alpaka::concepts::IView<double> auto offset,
            std::uint32_t const outputCount,
            std::uint32_t const seed,
            std::uint64_t const historyId) const
        {
            for(auto [index] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{1u}))
            {
                if(outputCount == 0u || totalWeight[0u] <= 0.0)
                {
                    offset[index] = 0.0;
                    return;
                }
                auto rng = alpaka::rand::engine::Philox4x32x10{seed, historyId};
                double const uniform = alpaka::rand::distribution::UniformReal<double>{}(rng);
                offset[index] = uniform * totalWeight[0u] / static_cast<double>(outputCount);
            }
        }
    };

    struct SelectParticleCombCandidates
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IView<double> auto const cdf,
            alpaka::concepts::IView<double> auto const totalWeight,
            alpaka::concepts::IView<double> auto const offset,
            alpaka::concepts::IView<std::uint32_t> auto selectedCandidates,
            std::uint32_t const candidateCount,
            std::uint32_t const outputCount,
            std::uint32_t const outputOffset) const
        {
            for(auto [output] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{outputCount}))
            {
                if(candidateCount == 0u || totalWeight[0u] <= 0.0)
                {
                    selectedCandidates[outputOffset + output] = candidateCount;
                    continue;
                }
                double const spacing = totalWeight[0u] / static_cast<double>(outputCount);
                double const threshold = offset[0u] + static_cast<double>(output) * spacing;
                std::uint32_t lower = 0u;
                std::uint32_t upper = candidateCount;
                while(lower < upper)
                {
                    std::uint32_t const middle = lower + (upper - lower) / 2u;
                    if(cdf[middle] <= threshold)
                        lower = middle + 1u;
                    else
                        upper = middle;
                }
                selectedCandidates[outputOffset + output] = lower < candidateCount ? lower : candidateCount - 1u;
            }
        }
    };

    /** @brief Persist the equal post-combing weight beside each selected candidate. */
    struct AssignParticleCombSelectedWeights
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IView<double> auto const totalWeight,
            alpaka::concepts::IView<double> auto selectedWeights,
            std::uint32_t const outputCount,
            std::uint32_t const outputOffset) const
        {
            for(auto [output] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{outputCount}))
                selectedWeights[outputOffset + output] = outputCount == 0u || totalWeight[0u] <= 0.0
                                                             ? 0.0
                                                             : totalWeight[0u] / static_cast<double>(outputCount);
        }
    };
} // namespace hase::kernels::forward
