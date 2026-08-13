// Copyright 2026 Tim Hanel
//
// This file is part of HASEonGPU
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <string_view>

#if HASE_ENABLE_ALPAKATUNE
#    include <alpakaTune/alpakaTune.hpp>
#    include <nlohmann/json.hpp>

#    include <algorithm>
#    include <array>
#    include <chrono>
#    include <cmath>
#    include <cstdint>
#    include <cstdlib>
#    include <filesystem>
#    include <fstream>
#    include <limits>
#    include <memory>
#    include <mutex>
#    include <optional>
#    include <stdexcept>
#    include <string>
#    include <unordered_map>
#    include <utility>
#    include <vector>
#endif

namespace hase::alpakaUtils
{
#if HASE_ENABLE_ALPAKATUNE
    namespace detail
    {
        inline auto requiredTimingEnvironment(char const* name) -> std::string
        {
            auto const* value = std::getenv(name);
            if(value == nullptr || value[0] == '\0')
                throw std::runtime_error{std::string{"Missing required environment variable: "} + name};
            return value;
        }

        inline auto optionalTimingEnvironment(char const* name) -> std::string
        {
            auto const* value = std::getenv(name);
            return value == nullptr ? std::string{} : std::string{value};
        }

        template<typename T_Vector>
        auto timingVectorJson(T_Vector const& value) -> nlohmann::json
        {
            auto result = nlohmann::json::array();
            for(std::size_t dimension = 0u; dimension < value.dim(); ++dimension)
                result.push_back(value[dimension]);
            return result;
        }

        class TimingTraceStore
        {
        public:
            ~TimingTraceStore() noexcept
            {
                try
                {
                    std::scoped_lock lock{m_mutex};
                    for(auto const& [pathText, records] : m_records)
                    {
                        auto const path = std::filesystem::path{pathText};
                        if(path.has_parent_path())
                            std::filesystem::create_directories(path.parent_path());
                        std::ofstream output{path, std::ios::app};
                        if(!output)
                            continue;
                        for(auto const& record : records)
                            output << record << '\n';
                    }
                }
                catch(...)
                {
                    // Static destruction cannot report trace persistence errors.
                }
            }

            void stage(std::filesystem::path const& path, nlohmann::json const& record)
            {
                std::scoped_lock lock{m_mutex};
                m_records[path.string()].push_back(record.dump());
            }

        private:
            std::mutex m_mutex;
            std::unordered_map<std::string, std::vector<std::string>> m_records;
        };

        inline auto timingTraceStore() -> TimingTraceStore&
        {
            static TimingTraceStore store;
            return store;
        }

        inline auto timingCampaignElapsedSeconds() -> double
        {
            static auto const started = std::chrono::steady_clock::now();
            return std::chrono::duration<double>{std::chrono::steady_clock::now() - started}.count();
        }

        inline auto tuningKernelEnabled(std::string_view kernel) -> bool
        {
            static auto const configured = optionalTimingEnvironment("HASE_ALPAKATUNE_KERNELS");
            if(configured.empty())
                return true;
            auto const entries = std::string_view{configured};
            auto begin = std::size_t{0u};
            while(begin <= entries.size())
            {
                auto const end = entries.find(',', begin);
                auto const entry = entries.substr(begin, end == std::string_view::npos ? end : end - begin);
                if(entry == kernel)
                    return true;
                if(end == std::string_view::npos)
                    break;
                begin = end + 1u;
            }
            return false;
        }
    } // namespace detail
#endif

    /**
     * Tune one selected one-dimensional kernel launch and independently
     * benchmark the actual launch with Alpaka events.
     *
     * Five legal candidates preserve the original worker coverage while using
     * 32, 64, 128, 256, or 512 workers per frame. Tuner measurements supply
     * search/adaptive durations; cached offline and fixed winners use the same
     * direct start/kernel/end event pattern without host recommendation time.
     */
    template<typename T_Queue, typename T_FrameSpec, typename T_Kernel, typename... T_Args>
    void timedEnqueue(
        T_Queue const& queue,
        T_FrameSpec const& frameSpec,
        alpaka::KernelBundle<T_Kernel, T_Args...> const& bundle,
        std::string_view kernelIdentity)
    {
#if HASE_ENABLE_ALPAKATUNE
        if(!detail::tuningKernelEnabled(kernelIdentity))
        {
            queue.enqueue(frameSpec, bundle);
            return;
        }
        if constexpr(
            !std::same_as<ALPAKA_TYPEOF(queue.getQueueKind()), alpaka::queueKind::NonBlocking>
            || !std::same_as<ALPAKA_TYPEOF(queue.getTiming()), alpaka::timing::Enabled>)
        {
            static_cast<void>(kernelIdentity);
            queue.enqueue(frameSpec, bundle);
            return;
        }
        else
        {
            using FrameExtents = ALPAKA_TYPEOF(frameSpec.getFrameExtents());
            using NumFrames = ALPAKA_TYPEOF(frameSpec.getNumFrames());
            static_assert(FrameExtents::dim() == 1u && NumFrames::dim() == 1u);
            using FrameScalar = ALPAKA_TYPEOF(frameSpec.getFrameExtents()[0u]);
            using FrameCountScalar = ALPAKA_TYPEOF(frameSpec.getNumFrames()[0u]);

            auto const coverage = static_cast<std::uintmax_t>(frameSpec.getFrameExtents()[0u])
                                  * static_cast<std::uintmax_t>(frameSpec.getNumFrames()[0u]);
            auto extentCandidates = std::vector<FrameExtents>{};
            auto frameCountCandidates = std::vector<NumFrames>{};
            for(auto const extentValue : std::array<std::uintmax_t, 5u>{32u, 64u, 128u, 256u, 512u})
            {
                if(coverage % extentValue != 0u || extentValue > std::numeric_limits<FrameScalar>::max()
                   || coverage / extentValue > std::numeric_limits<FrameCountScalar>::max())
                    continue;
                auto extent = frameSpec.getFrameExtents();
                auto frames = frameSpec.getNumFrames();
                extent[0u] = static_cast<FrameScalar>(extentValue);
                frames[0u] = static_cast<FrameCountScalar>(coverage / extentValue);
                extentCandidates.push_back(extent);
                frameCountCandidates.push_back(frames);
            }
            if(std::ranges::find(extentCandidates, frameSpec.getFrameExtents()) == extentCandidates.end())
                extentCandidates.push_back(frameSpec.getFrameExtents());
            if(std::ranges::find(frameCountCandidates, frameSpec.getNumFrames()) == frameCountCandidates.end())
                frameCountCandidates.push_back(frameSpec.getNumFrames());

            auto const frameExtentValues = alpakaTune::RVals<FrameExtents>{std::move(extentCandidates)};
            auto const numFrameValues = alpakaTune::RVals<NumFrames>{std::move(frameCountCandidates)};
            auto makeFrameTuning = [&]()
            {
                return alpakaTune::constrain(
                    alpakaTune::TunableBundle{
                        alpakaTune::tuneFrameExtent(frameSpec, frameExtentValues),
                        alpakaTune::tuneNumFrames(frameSpec, numFrameValues)},
                    alpakaTune::preserveCoverage(frameSpec));
            };
            using FrameTuning = ALPAKA_TYPEOF(makeFrameTuning());
            using Tuner = ALPAKA_TYPEOF(
                alpakaTune::makeTuner(
                    std::declval<alpakaTune::TunerConfig>(),
                    std::declval<FrameTuning>(),
                    queue.getDevice(),
                    frameSpec.getExecutor(),
                    kernelIdentity));
            using Device = std::remove_cvref_t<decltype(queue.getDevice())>;

            auto registryKey = std::string{kernelIdentity};
            auto appendShape = [&registryKey](std::string_view label, auto const& value)
            {
                registryKey += ':';
                registryKey += label;
                for(std::size_t dimension = 0u; dimension < value.dim(); ++dimension)
                    registryKey += ':' + std::to_string(value[dimension]);
            };
            appendShape("frames", frameSpec.getNumFrames());
            appendShape("extent", frameSpec.getFrameExtents());

            struct TunerEntry
            {
                explicit TunerEntry(Device value)
                    : device{std::move(value)}
                    , benchmarkStart{device.makeEvent(alpaka::timing::enabled)}
                    , benchmarkEnd{device.makeEvent(alpaka::timing::enabled)}
                {
                }

                Device device;
                ALPAKA_TYPEOF(std::declval<Device&>().makeEvent(alpaka::timing::enabled)) benchmarkStart;
                ALPAKA_TYPEOF(std::declval<Device&>().makeEvent(alpaka::timing::enabled)) benchmarkEnd;
                std::unique_ptr<Tuner> tuner;
                std::optional<alpakaTune::ParameterConfiguration> replayConfiguration;
                std::size_t replayCandidateIndex{};
                std::optional<std::string> learnedStatus;
                bool loadedFromCache{};
                std::mutex mutex;
            };

            static std::mutex registryMutex;
            static std::unordered_map<std::string, std::vector<std::shared_ptr<TunerEntry>>> registry;
            auto entry = std::shared_ptr<TunerEntry>{};
            {
                std::scoped_lock lock{registryMutex};
                auto& matchingEntries = registry[registryKey];
                auto const found = std::ranges::find_if(
                    matchingEntries,
                    [&queue](auto const& candidate) { return candidate->device == queue.getDevice(); });
                if(found != matchingEntries.end())
                    entry = *found;
                else
                {
                    entry = std::make_shared<TunerEntry>(queue.getDevice());
                    matchingEntries.push_back(entry);
                }
            }

            std::scoped_lock entryLock{entry->mutex};
            auto const benchmarkOnly = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_BENCHMARK_ONLY") == "1";
            if(!benchmarkOnly && !entry->tuner)
            {
                auto config
                    = alpakaTune::TunerConfig::fromYaml(detail::requiredTimingEnvironment("ALPAKA_TUNE_CONFIG"));
                if(auto const path = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_HISTORY"); !path.empty())
                    config.history.file = path;
                if(auto const path = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_COMPLETE_HISTORY");
                   !path.empty())
                    config.completeHistory.file = path;
                if(auto const model = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_MODEL"); !model.empty())
                    config.learnedModelFile = model;
                entry->tuner = std::make_unique<Tuner>(alpakaTune::makeTuner(
                    std::move(config),
                    makeFrameTuning(),
                    queue.getDevice(),
                    frameSpec.getExecutor(),
                    kernelIdentity));
            }

            auto selectVector = [](auto const& configuration, auto const& candidates, std::size_t configurationOffset)
            {
                ALPAKA_TYPEOF(candidates.front()) selected{};
                for(std::size_t dimension = 0u; dimension < selected.dim(); ++dimension)
                {
                    auto uniqueComponents = std::vector<ALPAKA_TYPEOF(candidates.front()[0u])>{};
                    for(auto const& candidate : candidates)
                    {
                        if(std::ranges::find(uniqueComponents, candidate[dimension]) == uniqueComponents.end())
                            uniqueComponents.push_back(candidate[dimension]);
                    }
                    auto const configurationIndex = configurationOffset + dimension;
                    auto const position = configuration.empty() || uniqueComponents.size() == 1u
                                              ? 0u
                                              : static_cast<std::size_t>(std::llround(
                                                    configuration.at(configurationIndex)
                                                    * static_cast<float>(uniqueComponents.size() - 1u)));
                    selected[dimension] = uniqueComponents.at(position);
                }
                return selected;
            };
            auto selectedExtent = frameSpec.getFrameExtents();
            auto selectedFrames = frameSpec.getNumFrames();
            auto observation = std::optional<alpakaTune::LaunchObservation>{};
            auto benchmarkRuntime = std::optional<double>{};
            auto measureDirect = [&](auto const& selectedFrameSpec)
            {
                queue.enqueue(entry->benchmarkStart);
                queue.enqueue(selectedFrameSpec, bundle);
                queue.enqueue(entry->benchmarkEnd);
                return alpaka::onHost::getElapsedTime(entry->benchmarkStart, entry->benchmarkEnd).count();
            };
            if(benchmarkOnly)
                benchmarkRuntime = measureDirect(frameSpec);
            else if(entry->replayConfiguration)
            {
                selectedExtent = selectVector(*entry->replayConfiguration, frameExtentValues.values(), 0u);
                selectedFrames
                    = selectVector(*entry->replayConfiguration, numFrameValues.values(), FrameExtents::dim());
                auto const selectedFrameSpec = T_FrameSpec{selectedFrames, selectedExtent, frameSpec.getExecutor()};
                benchmarkRuntime = measureDirect(selectedFrameSpec);
            }
            else
            {
                observation = entry->tuner->enqueueObserved(queue, frameSpec, bundle);
                selectedExtent = selectVector(observation->configuration, frameExtentValues.values(), 0u);
                selectedFrames
                    = selectVector(observation->configuration, numFrameValues.values(), FrameExtents::dim());
                entry->learnedStatus = observation->learnedStatus;
                entry->loadedFromCache = observation->loadedFromCache;
                if(observation->runtimeSeconds)
                    benchmarkRuntime = observation->runtimeSeconds;
                else if(observation->tuningComplete)
                {
                    // Cache the offline or terminal fixed winner selected by
                    // this unmeasured launch. Later calls use direct events.
                    entry->replayConfiguration = observation->configuration;
                    entry->replayCandidateIndex = observation->candidateIndex;
                }
            }
            if(auto const tracePath = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_TRACE"); !tracePath.empty())
            {
                auto record = nlohmann::json::object();
                record["kernel"] = kernelIdentity;
                record["mode"] = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_MODE");
                record["candidate_index"] = observation ? observation->candidateIndex : entry->replayCandidateIndex;
                record["configuration"] = observation                  ? nlohmann::json{observation->configuration}
                                          : benchmarkOnly              ? nlohmann::json::array()
                                          : entry->replayConfiguration ? nlohmann::json{*entry->replayConfiguration}
                                                                       : nlohmann::json::array();
                record["evaluation_runtime_seconds"] = benchmarkRuntime;
                record["evaluation_measured"] = benchmarkRuntime.has_value();
                record["evaluation_runtime_measurement_source"] = "device_event";
                record["tuner_runtime_seconds"] = observation ? observation->runtimeSeconds : std::nullopt;
                record["tuner_measured"] = observation && observation->measured;
                record["tuning_complete"]
                    = entry->replayConfiguration.has_value() || (observation && observation->tuningComplete);
                record["loaded_from_cache"] = observation ? observation->loadedFromCache : entry->loadedFromCache;
                record["learned_status"] = observation ? observation->learnedStatus : entry->learnedStatus;
                record["elapsed_seconds"] = detail::timingCampaignElapsedSeconds();
                record["original_num_frames"] = detail::timingVectorJson(frameSpec.getNumFrames());
                record["original_frame_extent"] = detail::timingVectorJson(frameSpec.getFrameExtents());
                record["selected_num_frames"] = detail::timingVectorJson(selectedFrames);
                record["selected_frame_extent"] = detail::timingVectorJson(selectedExtent);
                record["coverage"] = coverage;
                record["selected_coverage"] = static_cast<std::uintmax_t>(selectedFrames[0u])
                                              * static_cast<std::uintmax_t>(selectedExtent[0u]);
                record["candidate_count"] = benchmarkOnly ? 1u : entry->tuner->info().candidateCount;
                record["legal_candidate_count"] = frameExtentValues.size();
                detail::timingTraceStore().stage(tracePath, record);
            }
        }
#else
        static_cast<void>(kernelIdentity);
        queue.enqueue(frameSpec, bundle);
#endif
    }
} // namespace hase::alpakaUtils
