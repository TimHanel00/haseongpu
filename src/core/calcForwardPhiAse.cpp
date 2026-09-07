/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <alpaka/math.hpp>

#include <core/calcForwardPhiAse.hpp>
#include <kernels/forward/rayPopulationStatistics.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace hase::core
{
    ForwardPhiAseRawResult makeForwardRawResult(
        unsigned const volumeCount,
        unsigned const materialVertexCount,
        unsigned const numIndependentRayPopulations)
    {
        if(numIndependentRayPopulations == 0u)
            throw std::invalid_argument("forward ASE batch count must be positive");
        return ForwardPhiAseRawResult{
            std::vector<double>(static_cast<std::size_t>(numIndependentRayPopulations) * materialVertexCount, 0.0),
            std::vector<unsigned>(numIndependentRayPopulations, 0u),
            std::vector<unsigned>(volumeCount, 0u),
            std::vector<unsigned>(volumeCount, 0u),
            0u,
            data::BoundaryStatus::disabled,
            0u,
            0.0,
            0u,
            0u,
            0.0,
            0.0,
            0.0,
            0.0};
    }

    double calcForwardSourceStrengthTotal(hase::data::TraceData const& trace)
    {
        return trace.sourceStrengthPrefix.empty() ? 0.0 : trace.sourceStrengthPrefix.back();
    }

    void mergeForwardRawResult(ForwardPhiAseRawResult& target, ForwardPhiAseRawResult const& source)
    {
        if(target.vertexPopulationScoreSum.empty())
        {
            target = source;
            return;
        }

        target.rayCount += source.rayCount;
        if(boundaryStatusPriority(source.boundaryStatus) > boundaryStatusPriority(target.boundaryStatus))
            target.boundaryStatus = source.boundaryStatus;
        target.boundaryPasses = std::max(target.boundaryPasses, source.boundaryPasses);
        target.boundaryRemainingFraction
            = std::max(target.boundaryRemainingFraction, source.boundaryRemainingFraction);
        target.boundaryMaxPasses = std::max(target.boundaryMaxPasses, source.boundaryMaxPasses);
        target.boundaryDivergenceStreak = std::max(target.boundaryDivergenceStreak, source.boundaryDivergenceStreak);
        target.boundaryGamma = std::max(target.boundaryGamma, source.boundaryGamma);
        target.boundaryGammaStandardError
            = std::max(target.boundaryGammaStandardError, source.boundaryGammaStandardError);
        target.boundaryTailFactor = std::max(target.boundaryTailFactor, source.boundaryTailFactor);
        target.boundaryTailClosure = std::max(target.boundaryTailClosure, source.boundaryTailClosure);
        if(target.vertexPopulationScoreSum.size() != source.vertexPopulationScoreSum.size())
            throw std::runtime_error("cannot merge forward ASE results with different vertex counts");
        for(unsigned vertex = 0u; vertex < target.vertexPopulationScoreSum.size(); ++vertex)
            target.vertexPopulationScoreSum.at(vertex) += source.vertexPopulationScoreSum.at(vertex);
        if(target.rayPopulationRayCounts.size() != source.rayPopulationRayCounts.size())
            throw std::runtime_error("cannot merge forward ASE results with different batch counts");
        for(unsigned batch = 0u; batch < target.rayPopulationRayCounts.size(); ++batch)
            target.rayPopulationRayCounts.at(batch) += source.rayPopulationRayCounts.at(batch);
        for(unsigned volume = 0u; volume < target.totalRays.size(); ++volume)
        {
            target.totalRays.at(volume) += source.totalRays.at(volume);
            target.droppedRays.at(volume) += source.droppedRays.at(volume);
        }
    }

    double calcForwardRelativeStandardError(
        double const scoreSum,
        double const scoreSquareSum,
        unsigned const rayCount)
    {
        if(rayCount < 2u || !alpaka::math::isfinite(scoreSum) || !alpaka::math::isfinite(scoreSquareSum))
        {
            return std::numeric_limits<double>::max();
        }
        if(scoreSum == 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double const n = static_cast<double>(rayCount);
        double const relativeVariance = (n * scoreSquareSum / (scoreSum * scoreSum) - 1.0) / n;
        return std::sqrt(std::max(0.0, relativeVariance));
    }

    double calcForwardStandardError(
        double const scoreSum,
        double const scoreSquareSum,
        unsigned const rayCount,
        double const normalizationVolume,
        double const volumeSize)
    {
        if(rayCount < 2u || volumeSize <= 0.0 || normalizationVolume < 0.0 || !alpaka::math::isfinite(scoreSum)
           || !alpaka::math::isfinite(scoreSquareSum))
        {
            return std::numeric_limits<double>::max();
        }

        double const relativeStandardError = calcForwardRelativeStandardError(scoreSum, scoreSquareSum, rayCount);
        if(alpaka::math::isnan(relativeStandardError))
        {
            return 0.0;
        }
        if(!alpaka::math::isfinite(relativeStandardError))
        {
            return std::numeric_limits<double>::max();
        }

        double const volumeScale = normalizationVolume / volumeSize;
        double const estimate = scoreSum * volumeScale / rayCount;
        return relativeStandardError * std::abs(estimate);
    }

    void finalizeForwardPhiAse(
        hase::data::TraceData const& hostMesh,
        ForwardPhiAseRawResult const& rawResult,
        data::PhiAseResult& result)
    {
        finalizeForwardPhiAse(hostMesh, rawResult, calcForwardSourceStrengthTotal(hostMesh), result);
    }

    void finalizeForwardPhiAse(
        hase::data::TraceData const& hostMesh,
        ForwardPhiAseRawResult const& rawResult,
        double const sourceStrengthTotal,
        data::PhiAseResult& result)
    {
        unsigned const volumeCount = hostMesh.numberOfCells;
        std::size_t const materialVertexCount
            = static_cast<std::size_t>(hostMesh.numberOfMaterials) * hostMesh.numberOfMeshPoints;
        unsigned const numIndependentRayPopulations = static_cast<unsigned>(rawResult.rayPopulationRayCounts.size());
        if(numIndependentRayPopulations == 0u
           || rawResult.vertexPopulationScoreSum.size() != numIndependentRayPopulations * materialVertexCount)
            throw std::runtime_error("forward ASE vertex score count does not match the mesh");
        std::vector<std::vector<double>> cellBatchScoreDensity(numIndependentRayPopulations);
        for(unsigned batch = 0u; batch < numIndependentRayPopulations; ++batch)
        {
            auto const begin = rawResult.vertexPopulationScoreSum.cbegin() + batch * materialVertexCount;
            std::vector<double> const vertexBatch(begin, begin + materialVertexCount);
            cellBatchScoreDensity.at(batch)
                = hase::kernels::accumulateMaterialVertexIntegralsToCellDensities(hostMesh, vertexBatch);
        }
        result = data::PhiAseResult(
            std::vector(volumeCount, 0.0f),
            std::vector(volumeCount, 0.0),
            std::vector(volumeCount, 0.0),
            rawResult.totalRays,
            std::vector(volumeCount, 0.0),
            rawResult.droppedRays,
            rawResult.boundaryStatus,
            rawResult.boundaryPasses,
            rawResult.boundaryRemainingFraction,
            rawResult.boundaryMaxPasses,
            rawResult.boundaryDivergenceStreak,
            rawResult.boundaryGamma,
            rawResult.boundaryGammaStandardError,
            rawResult.boundaryTailFactor,
            rawResult.boundaryTailClosure);
        for(unsigned volume = 0u; volume < volumeCount; ++volume)
        {
            double const volumeSize = hostMesh.cellVolumes.at(volume);
            if(volumeSize > 0.0 && rawResult.rayCount > 0u)
            {
                kernels::forward::RayPopulationStatistics statistics;
                for(unsigned batch = 0u; batch < numIndependentRayPopulations; ++batch)
                    statistics.add(
                        cellBatchScoreDensity.at(batch).at(volume),
                        rawResult.rayPopulationRayCounts.at(batch));
                auto const summary = statistics.finalize(sourceStrengthTotal, result.droppedRays[volume] != 0u);
                double const estimate = summary.value;
                result.phiAse.at(volume) = static_cast<float>(estimate);
                result.relativeStandardError.at(volume) = summary.relativeStandardError;
                result.standardError.at(volume) = summary.standardError;
                unsigned const material = hostMesh.cellMaterialIds.at(volume);
                double const gainPerDensity = hostMesh.materialActive.at(material) != 0u
                                                  ? hostMesh.betaVolume.at(volume)
                                                            * (hostMesh.materialPeakEmission.at(material)
                                                               + hostMesh.materialPeakAbsorption.at(material))
                                                        - hostMesh.materialPeakAbsorption.at(material)
                                                  : 0.0;
                result.dndtAse.at(volume) = gainPerDensity * estimate;
            }
            else
            {
                result.phiAse.at(volume) = 0.0f;
                result.standardError.at(volume) = std::numeric_limits<double>::max();
                result.relativeStandardError.at(volume) = std::numeric_limits<double>::max();
            }
        }
    }
} // namespace hase::core
