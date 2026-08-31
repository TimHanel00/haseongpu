#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <span>

namespace hase::kernels::forward
{
    struct SelectBoundaryRouteWeight
    {
        std::uint32_t domain;

        ALPAKA_FN_ACC double operator()(double const weight, std::uint32_t const targetDomain) const
        {
            return targetDomain == domain ? weight : 0.0;
        }
    };

    struct SelectBoundaryRouteCandidate
    {
        std::uint32_t domain;

        ALPAKA_FN_ACC std::uint32_t operator()(double const weight, std::uint32_t const targetDomain) const
        {
            return targetDomain == domain && weight > 0.0 ? 1u : 0u;
        }
    };

    struct MakeBoundarySpatialKeys
    {
        [[nodiscard]] ALPAKA_FN_ACC static std::uint64_t quantize(double const value)
        {
            double const clamped = value < 0.0 ? 0.0 : (value > 1.0 ? 1.0 : value);
            return static_cast<std::uint64_t>(clamped * 16383.0);
        }

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto const weights,
            auto const targetDomains,
            auto const targetCells,
            auto const targetFaces,
            auto const barycentricX,
            auto const barycentricY,
            auto keys,
            auto indices,
            std::uint32_t const candidateCount,
            std::uint32_t const domain) const
        {
            for(auto [candidate] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{candidateCount}))
            {
                indices[candidate] = candidate;
                if(targetDomains[candidate] != domain || weights[candidate] <= 0.0)
                {
                    keys[candidate] = std::numeric_limits<std::uint64_t>::max();
                    continue;
                }
                keys[candidate] = (static_cast<std::uint64_t>(targetCells[candidate]) << 32u)
                                  | (static_cast<std::uint64_t>(targetFaces[candidate] & 0x0fu) << 28u)
                                  | (quantize(barycentricX[candidate]) << 14u) | quantize(barycentricY[candidate]);
            }
        }
    };

    struct MarkBoundaryRadixZeros
    {
        std::uint32_t bit;

        ALPAKA_FN_ACC std::uint32_t operator()(std::uint64_t const key) const
        {
            return ((key >> bit) & 1u) == 0u ? 1u : 0u;
        }
    };

    struct CaptureBoundaryRadixZeroCount
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto const flags,
            auto const prefix,
            auto zeroCount,
            std::uint32_t const candidateCount) const
        {
            for(auto [index] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{1u}))
                zeroCount[index]
                    = candidateCount == 0u ? 0u : prefix[candidateCount - 1u] + flags[candidateCount - 1u];
        }
    };

    struct ScatterBoundaryRadixPass
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto const inputKeys,
            auto const inputIndices,
            auto const flags,
            auto const prefix,
            auto const zeroCount,
            auto outputKeys,
            auto outputIndices,
            std::uint32_t const candidateCount) const
        {
            for(auto [candidate] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{candidateCount}))
            {
                auto const destination
                    = flags[candidate] != 0u ? prefix[candidate] : zeroCount[0u] + candidate - prefix[candidate];
                outputKeys[destination] = inputKeys[candidate];
                outputIndices[destination] = inputIndices[candidate];
            }
        }
    };

    struct GatherBoundaryRouteWeights
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto const candidateWeights,
            auto const sortedIndices,
            auto sortedWeights,
            std::uint32_t const routeCandidateCount) const
        {
            for(auto [candidate] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{routeCandidateCount}))
                sortedWeights[candidate] = candidateWeights[sortedIndices[candidate]];
        }
    };

    struct SelectSpatialBoundaryCandidates
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto const cdf,
            auto const sortedIndices,
            auto selectedCandidates,
            auto selectedPositions,
            std::uint32_t const routeCandidateCount,
            std::uint32_t const outputOffset,
            std::uint32_t const outputCount,
            std::uint32_t const seed,
            std::uint64_t const historyId) const
        {
            for(auto [output] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{outputCount}))
            {
                std::uint32_t const begin = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(output) * routeCandidateCount / outputCount);
                std::uint32_t const end = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(output + 1u) * routeCandidateCount / outputCount);
                double const lowerWeight = begin == 0u ? 0.0 : cdf[begin - 1u];
                double const upperWeight = cdf[end - 1u];
                auto rng = alpaka::rand::engine::Philox4x32x10{seed, historyId + output};
                double const uniform = alpaka::rand::distribution::UniformReal<double>{}(rng);
                double const threshold = lowerWeight + uniform * (upperWeight - lowerWeight);
                std::uint32_t lower = begin;
                std::uint32_t upper = end;
                while(lower < upper)
                {
                    auto const middle = lower + (upper - lower) / 2u;
                    if(cdf[middle] <= threshold)
                        lower = middle + 1u;
                    else
                        upper = middle;
                }
                auto const position = lower < end ? lower : end - 1u;
                selectedPositions[outputOffset + output] = position;
                selectedCandidates[outputOffset + output] = sortedIndices[position];
            }
        }
    };

    struct RedistributeBoundaryRouteWeights
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            auto const candidateWeights,
            auto const sortedIndices,
            auto const selectedPositions,
            auto const selectedCandidates,
            auto const positionX,
            auto const positionY,
            auto const positionZ,
            auto selectedWeights,
            std::uint32_t const routeCandidateCount,
            std::uint32_t const outputOffset,
            std::uint32_t const outputCount) const
        {
            for(auto [position] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{routeCandidateCount}))
            {
                std::uint32_t lower = 0u;
                std::uint32_t upper = outputCount;
                while(lower < upper)
                {
                    auto const middle = lower + (upper - lower) / 2u;
                    if(selectedPositions[outputOffset + middle] < position)
                        lower = middle + 1u;
                    else
                        upper = middle;
                }
                double const weight = candidateWeights[sortedIndices[position]];
                if(lower < outputCount && selectedPositions[outputOffset + lower] == position)
                {
                    alpaka::onAcc::atomicAdd(acc, &selectedWeights[outputOffset + lower], weight);
                    continue;
                }
                if(lower == 0u)
                {
                    alpaka::onAcc::atomicAdd(acc, &selectedWeights[outputOffset], weight);
                    continue;
                }
                if(lower == outputCount)
                {
                    alpaka::onAcc::atomicAdd(acc, &selectedWeights[outputOffset + outputCount - 1u], weight);
                    continue;
                }
                auto const leftPosition = selectedPositions[outputOffset + lower - 1u];
                auto const rightPosition = selectedPositions[outputOffset + lower];
                auto const candidate = sortedIndices[position];
                auto const leftCandidate = selectedCandidates[outputOffset + lower - 1u];
                auto const rightCandidate = selectedCandidates[outputOffset + lower];
                double const segmentX = positionX[rightCandidate] - positionX[leftCandidate];
                double const segmentY = positionY[rightCandidate] - positionY[leftCandidate];
                double const segmentZ = positionZ[rightCandidate] - positionZ[leftCandidate];
                double const segmentNorm = segmentX * segmentX + segmentY * segmentY + segmentZ * segmentZ;
                double const projection = segmentNorm > 0.0
                                              ? ((positionX[candidate] - positionX[leftCandidate]) * segmentX
                                                 + (positionY[candidate] - positionY[leftCandidate]) * segmentY
                                                 + (positionZ[candidate] - positionZ[leftCandidate]) * segmentZ)
                                                    / segmentNorm
                                              : 0.5;
                double const rightFraction = projection < 0.0 ? 0.0 : (projection > 1.0 ? 1.0 : projection);
                alpaka::onAcc::atomicAdd(
                    acc,
                    &selectedWeights[outputOffset + lower - 1u],
                    weight * (1.0 - rightFraction));
                alpaka::onAcc::atomicAdd(acc, &selectedWeights[outputOffset + lower], weight * rightFraction);
            }
        }
    };

    struct MarkLiveBoundaryParents
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IView<double> auto const candidateWeights,
            alpaka::concepts::IView<std::uint32_t> auto liveParents,
            std::uint32_t const parentCount) const
        {
            for(auto [parent] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{parentCount}))
                liveParents[parent]
                    = candidateWeights[2u * parent] > 0.0 || candidateWeights[2u * parent + 1u] > 0.0 ? 1u : 0u;
        }
    };

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
