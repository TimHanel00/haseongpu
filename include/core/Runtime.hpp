/**
 * Copyright 2015 Erik Zenker, Carlchristian Eckert
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <alpaka/math.hpp>

#include <data/PhiAseResult.hpp>
#include <data/TraceData.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace hase::core
{
    namespace fs = std::filesystem;

    /** @brief Stable names for the supported host-level execution strategies. */
    struct ParallelMode
    {
        static inline std::string const NONE = "no_parallel_mode";
        static inline std::string const SINGLE = "single";
        static inline std::string const MPI = "mpi";
    };

    /** @brief Host scheduling policy kept separate from physical trace data. */
    struct ExecutionPolicy
    {
        static constexpr unsigned unspecifiedRngSeed = std::numeric_limits<unsigned>::max();

        ExecutionPolicy()
        {
        }

        ExecutionPolicy(
            unsigned maxRepetitions,
            unsigned adaptiveSteps,
            unsigned numDevices,
            unsigned gpu_i,
            std::string backend,
            std::string parallelMode,
            bool writeVtk,
            std::vector<unsigned> devices,
            unsigned minSampleRange,
            unsigned maxSampleRange,
            unsigned rngSeed = unspecifiedRngSeed)
            : maxRepetitions(maxRepetitions)
            , adaptiveSteps(adaptiveSteps)
            , numDevices(numDevices)
            , gpu_i(gpu_i)
            , backend(std::move(backend))
            , parallelMode(std::move(parallelMode))
            , writeVtk(writeVtk)
            , devices(std::move(devices))
            , minSampleRange(minSampleRange)
            , maxSampleRange(maxSampleRange)
            , rngSeed(rngSeed)
        {
        }

        ExecutionPolicy(
            unsigned maxRepetitions,
            unsigned adaptiveSteps,
            unsigned gpu_i,
            std::string backend,
            std::string parallelMode,
            bool writeVtk,
            fs::path inputPath,
            fs::path outputPath,
            std::vector<unsigned> devices,
            unsigned minSampleRange,
            unsigned maxSampleRange,
            unsigned numDevices,
            unsigned rngSeed = unspecifiedRngSeed)
            : maxRepetitions(maxRepetitions)
            , adaptiveSteps(adaptiveSteps)
            , numDevices(numDevices)
            , gpu_i(gpu_i)
            , backend(std::move(backend))
            , parallelMode(std::move(parallelMode))
            , writeVtk(writeVtk)
            , inputPath(std::move(inputPath))
            , outputPath(std::move(outputPath))
            , devices(std::move(devices))
            , minSampleRange(minSampleRange)
            , maxSampleRange(maxSampleRange)
            , rngSeed(rngSeed)
        {
        }

        unsigned maxRepetitions;
        unsigned adaptiveSteps;
        // user defined nr of gpus
        unsigned numDevices;
        unsigned gpu_i;
        std::string backend;
        std::string parallelMode;
        bool writeVtk;
        fs::path inputPath;
        fs::path outputPath;
        // gpu ids from cuda api
        std::vector<unsigned> devices;
        unsigned minSampleRange;
        unsigned maxSampleRange;
        unsigned rngSeed = unspecifiedRngSeed;
    };

    /** @brief Observed process and device layout reported with trace results. */
    struct RuntimeTopology
    {
        unsigned activeNodes = 1;
        unsigned activeRanks = 1;
        unsigned activeGpus = 0;
        double avgActiveRanksPerNode = 1.0;
        unsigned minActiveRanksPerNode = 1;
        unsigned maxActiveRanksPerNode = 1;
        double avgGpusPerRank = 0.0;
        double avgGpusPerNode = 0.0;
        unsigned minGpusPerNode = 0;
        unsigned maxGpusPerNode = 0;
    };

    /** @brief Physical and statistical controls for one domain-local ASE trace. */
    struct AseTraceControls
    {
        AseTraceControls() = default;

        [[nodiscard]] bool isForwardPropagation() const
        {
            return propagationMode == "forward";
        }

        [[nodiscard]] unsigned resolvedForwardRayCount() const
        {
            return forwardRayCount == 0u ? minRays : forwardRayCount;
        }

        unsigned minRays = 0u;
        unsigned maxRays = 0u;
        unsigned forwardRayCount = 0u;
        std::string propagationMode = "forward";
        double relativeStandardErrorThreshold = 0.0;
        bool useReflections = false;
        bool monochromatic = false;
        unsigned reflectionMaxIterations = 40u;
        double reflectionTolerance = 1.0e-4;
        unsigned surfaceReservoirSize = 32u;
    };

    [[nodiscard]] inline unsigned adaptiveRayTarget(
        AseTraceControls const& experiment,
        ExecutionPolicy const& compute,
        unsigned const completedIncreases)
    {
        if(experiment.forwardRayCount != 0u || experiment.maxRays <= experiment.minRays || compute.adaptiveSteps == 0u)
        {
            return experiment.resolvedForwardRayCount();
        }
        if(completedIncreases >= compute.adaptiveSteps)
        {
            return experiment.maxRays;
        }

        double const growth = std::pow(
            static_cast<double>(experiment.maxRays) / static_cast<double>(experiment.minRays),
            1.0 / static_cast<double>(compute.adaptiveSteps));
        double const target
            = static_cast<double>(experiment.minRays) * std::pow(growth, static_cast<double>(completedIncreases));
        unsigned const rounded = static_cast<unsigned>(std::ceil(target));
        return std::clamp(rounded, experiment.minRays, experiment.maxRays);
    }

    [[nodiscard]] inline bool forwardResultMeetsRelativeStandardError(
        data::PhiAseResult const& result,
        double const relativeStandardErrorThreshold)
    {
        return !result.relativeStandardError.empty()
               && std::all_of(
                   result.relativeStandardError.cbegin(),
                   result.relativeStandardError.cend(),
                   [relativeStandardErrorThreshold](double const relativeStandardError)
                   {
                       return alpaka::math::isfinite(relativeStandardError)
                              && relativeStandardError <= relativeStandardErrorThreshold;
                   });
    }

    inline void recordAdaptiveRayConvergence(
        data::PhiAseResult const& result,
        unsigned const targetRayCount,
        double const relativeStandardErrorThreshold,
        std::vector<unsigned>& convergenceRayCounts)
    {
        if(convergenceRayCounts.empty())
        {
            convergenceRayCounts.assign(result.relativeStandardError.size(), 0u);
        }

        unsigned const volumeCount = std::min(
            static_cast<unsigned>(convergenceRayCounts.size()),
            static_cast<unsigned>(result.relativeStandardError.size()));
        for(unsigned volume = 0u; volume < volumeCount; ++volume)
        {
            bool const hasDroppedRays = !result.droppedRays.empty() && result.droppedRays.at(volume) != 0u;
            double const relativeStandardError = result.relativeStandardError.at(volume);
            if(convergenceRayCounts.at(volume) == 0u && !hasDroppedRays
               && alpaka::math::isfinite(relativeStandardError)
               && relativeStandardError <= relativeStandardErrorThreshold)
            {
                convergenceRayCounts.at(volume) = targetRayCount;
            }
        }
    }


} // namespace hase::core
