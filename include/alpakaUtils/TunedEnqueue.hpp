// Copyright 2026 Tim Hanel
//
// This file is part of HASEonGPU
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <string_view>

#if HASE_ENABLE_ALPAKATUNE
#    include <alpakaTune/alpakaTune.hpp>
#    include <nlohmann/json.hpp>

#    include <algorithm>
#    include <chrono>
#    include <cmath>
#    include <cstdlib>
#    include <filesystem>
#    include <fstream>
#    include <memory>
#    include <mutex>
#    include <stdexcept>
#    include <string>
#    include <unordered_map>
#    include <utility>
#    include <vector>
#endif

namespace hase::alpakaUtils
{
    struct TunedLaunchObservation
    {
        std::size_t coverage{};
    };

#if HASE_ENABLE_ALPAKATUNE
    namespace detail
    {
        class TraceStore
        {
        public:
            ~TraceStore() noexcept
            {
                try
                {
                    std::scoped_lock lock{m_mutex};
                    for(auto const& [pathText, records] : m_records)
                    {
                        auto const path = std::filesystem::path{pathText};
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
                    // Process shutdown cannot report trace persistence failures.
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

        inline auto traceStore() -> TraceStore&
        {
            static TraceStore store;
            return store;
        }

        struct Metrics
        {
            std::size_t invocationCount{};
            std::size_t measuredCount{};
            double hostCallSeconds{};
            double measuredHostCallSeconds{};
            double measuredKernelSeconds{};
            double recommendationSeconds{};
            double measuredRecommendationSeconds{};
        };

        class MetricsStore
        {
        public:
            ~MetricsStore() noexcept
            {
                try
                {
                    auto records = Records{};
                    for(auto const& shard : m_shards)
                    {
                        for(auto const& [pathText, kernels] : shard->records)
                        {
                            for(auto const& [kernel, metrics] : kernels)
                            {
                                auto& target = records[pathText][kernel];
                                target.invocationCount += metrics.invocationCount;
                                target.measuredCount += metrics.measuredCount;
                                target.hostCallSeconds += metrics.hostCallSeconds;
                                target.measuredHostCallSeconds += metrics.measuredHostCallSeconds;
                                target.measuredKernelSeconds += metrics.measuredKernelSeconds;
                                target.recommendationSeconds += metrics.recommendationSeconds;
                                target.measuredRecommendationSeconds += metrics.measuredRecommendationSeconds;
                            }
                        }
                    }
                    for(auto const& [pathText, kernels] : records)
                    {
                        auto const path = std::filesystem::path{pathText};
                        std::filesystem::create_directories(path.parent_path());
                        std::ofstream output{path, std::ios::app};
                        if(!output)
                            continue;
                        for(auto const& [kernel, metrics] : kernels)
                        {
                            auto record = nlohmann::json::object();
                            record["kernel"] = kernel;
                            record["invocation_count"] = metrics.invocationCount;
                            record["measured_count"] = metrics.measuredCount;
                            record["replay_count"] = metrics.invocationCount - metrics.measuredCount;
                            record["host_call_seconds"] = metrics.hostCallSeconds;
                            record["measured_host_call_seconds"] = metrics.measuredHostCallSeconds;
                            record["measured_kernel_seconds"] = metrics.measuredKernelSeconds;
                            record["recommendation_seconds"] = metrics.recommendationSeconds;
                            record["measured_recommendation_seconds"] = metrics.measuredRecommendationSeconds;
                            record["estimated_measured_control_and_sync_seconds"]
                                = metrics.measuredHostCallSeconds - metrics.measuredKernelSeconds
                                  - metrics.measuredRecommendationSeconds;
                            output << record.dump() << '\n';
                        }
                    }
                }
                catch(...)
                {
                    // Process shutdown cannot report metrics persistence failures.
                }
            }

            template<typename T_Observation>
            void record(
                std::filesystem::path const& path,
                std::string_view kernel,
                T_Observation const& observation,
                double hostCallSeconds)
            {
                auto& metrics = shardForCurrentThread().records[path.string()][std::string{kernel}];
                ++metrics.invocationCount;
                metrics.hostCallSeconds += hostCallSeconds;
                metrics.recommendationSeconds += observation.recommendationSeconds;
                if(observation.measured)
                {
                    ++metrics.measuredCount;
                    metrics.measuredHostCallSeconds += hostCallSeconds;
                    metrics.measuredRecommendationSeconds += observation.recommendationSeconds;
                    if(observation.runtimeSeconds)
                        metrics.measuredKernelSeconds += *observation.runtimeSeconds;
                }
            }

        private:
            using Records = std::unordered_map<std::string, std::unordered_map<std::string, Metrics>>;

            struct Shard
            {
                Records records;
            };

            auto registerShard() -> std::shared_ptr<Shard>
            {
                auto shard = std::make_shared<Shard>();
                std::scoped_lock lock{m_mutex};
                m_shards.push_back(shard);
                return shard;
            }

            auto shardForCurrentThread() -> Shard&
            {
                static thread_local auto shard = registerShard();
                return *shard;
            }

            std::mutex m_mutex;
            std::vector<std::shared_ptr<Shard>> m_shards;
        };

        inline auto metricsStore() -> MetricsStore&
        {
            static MetricsStore store;
            return store;
        }

        inline auto requiredEnvironment(char const* name) -> std::string
        {
            auto const* value = std::getenv(name);
            if(value == nullptr || value[0] == '\0')
                throw std::runtime_error{std::string{"Missing required environment variable: "} + name};
            return value;
        }

        inline auto optionalEnvironment(char const* name) -> std::string
        {
            auto const* value = std::getenv(name);
            return value == nullptr ? std::string{} : std::string{value};
        }

        inline auto kernelTuningEnabled(std::string_view kernel) -> bool
        {
            static auto const configured = optionalEnvironment("HASE_ALPAKATUNE_KERNELS");
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

        inline void appendTrace(nlohmann::json const& record)
        {
            auto const path = std::filesystem::path{requiredEnvironment("HASE_ALPAKATUNE_TRACE")};
            traceStore().stage(path, record);
        }

        template<typename T_Observation>
        inline void appendMetrics(
            std::string_view kernel,
            T_Observation const& observation,
            double hostCallSeconds)
        {
            static auto const pathText = optionalEnvironment("HASE_ALPAKATUNE_METRICS");
            if(!pathText.empty())
                metricsStore().record(
                    std::filesystem::path{pathText},
                    kernel,
                    observation,
                    hostCallSeconds);
        }

        inline auto campaignElapsedSeconds() -> double
        {
            static auto const started = std::chrono::steady_clock::now();
            return std::chrono::duration<double>{std::chrono::steady_clock::now() - started}.count();
        }

        template<typename T_Vector>
        auto vectorJson(T_Vector const& value) -> nlohmann::json
        {
            auto result = nlohmann::json::array();
            for(std::size_t dimension = 0u; dimension < value.dim(); ++dimension)
                result.push_back(value[dimension]);
            return result;
        }
    } // namespace detail
#endif

    template<typename T_Queue, typename T_FrameSpec, typename T_Kernel, typename... T_Args>
    auto tunedEnqueue(
        T_Queue const& queue,
        T_FrameSpec const& frameSpec,
        alpaka::KernelBundle<T_Kernel, T_Args...> const& bundle,
        std::string_view kernelIdentity) -> TunedLaunchObservation
    {
        auto const coverage = static_cast<std::size_t>(frameSpec.getNumFrames().product())
                              * static_cast<std::size_t>(frameSpec.getFrameExtents().product());
#if HASE_ENABLE_ALPAKATUNE
        if(!detail::kernelTuningEnabled(kernelIdentity))
        {
            queue.enqueue(frameSpec, bundle);
            return TunedLaunchObservation{coverage};
        }

        using FrameExtents = std::remove_cvref_t<decltype(frameSpec.getFrameExtents())>;
        using NumFrames = std::remove_cvref_t<decltype(frameSpec.getNumFrames())>;
        auto extents = std::vector<FrameExtents>{frameSpec.getFrameExtents()};
        auto alternateExtent = frameSpec.getFrameExtents();
        auto hasAlternate = false;
        for(std::size_t dimension = 0u; dimension < alternateExtent.dim(); ++dimension)
        {
            if(alternateExtent[dimension] > 1u && alternateExtent[dimension] % 2u == 0u)
            {
                alternateExtent[dimension] /= 2u;
                hasAlternate = true;
            }
        }
        auto const baselineOnly = detail::optionalEnvironment("HASE_ALPAKATUNE_BASELINE_ONLY") == "1";
        if(hasAlternate && !baselineOnly)
            extents.push_back(alternateExtent);

        auto makeFrameTuning = [&]()
        {
            if(baselineOnly)
            {
                return alpakaTune::makeFrameSpecTuning(
                    alpakaTune::tuneFrameExtent(
                        frameSpec,
                        alpakaTune::RVals<FrameExtents>{std::vector<FrameExtents>{frameSpec.getFrameExtents()}}),
                    alpakaTune::tuneNumFrames(
                        frameSpec,
                        alpakaTune::RVals<NumFrames>{std::vector<NumFrames>{frameSpec.getNumFrames()}}),
                    alpakaTune::preserveCoverage(frameSpec));
            }
            return alpakaTune::makeFrameSpecTuning(frameSpec);
        };
        using FrameTuning = decltype(makeFrameTuning());
        using Tuner = decltype(alpakaTune::makeTuner(
            std::declval<alpakaTune::TunerConfig>(),
            std::declval<FrameTuning>(),
            queue.getDevice(),
            frameSpec.getExecutor(),
            kernelIdentity));

        auto registryKey = std::string{kernelIdentity} + ":" + std::to_string(coverage)
                           + (baselineOnly ? ":baseline" : ":tunable");
        auto appendShape = [&registryKey](std::string_view label, auto const& value)
        {
            registryKey += ":";
            registryKey += label;
            for(std::size_t dimension = 0u; dimension < value.dim(); ++dimension)
                registryKey += ":" + std::to_string(value[dimension]);
        };
        appendShape("frames", frameSpec.getNumFrames());
        appendShape("extent", frameSpec.getFrameExtents());

        using Device = std::remove_cvref_t<decltype(queue.getDevice())>;
        struct TunerEntry
        {
            explicit TunerEntry(Device value) : device{std::move(value)}
            {
            }

            Device device;
            std::unique_ptr<Tuner> tuner;
            bool completionTraced{};
            std::mutex mutex;
        };
        static std::mutex tunerRegistryMutex;
        static std::unordered_map<std::string, std::vector<std::shared_ptr<TunerEntry>>> tuners;
        auto entry = std::shared_ptr<TunerEntry>{};
        {
            std::scoped_lock lock{tunerRegistryMutex};
            auto& matchingEntries = tuners[registryKey];
            auto const found = std::ranges::find_if(
                matchingEntries,
                [&queue](auto const& candidate)
                {
                    return candidate->device == queue.getDevice();
                });
            if(found != matchingEntries.end())
                entry = *found;
            else
            {
                entry = std::make_shared<TunerEntry>(queue.getDevice());
                matchingEntries.push_back(entry);
            }
        }

        std::scoped_lock entryLock{entry->mutex};
        if(!entry->tuner)
        {
            auto config = alpakaTune::TunerConfig::fromYaml(detail::requiredEnvironment("ALPAKA_TUNE_CONFIG"));
            config.history.file = detail::requiredEnvironment("HASE_ALPAKATUNE_HISTORY");
            if(auto const model = detail::optionalEnvironment("HASE_ALPAKATUNE_MODEL"); !model.empty())
                config.learnedModelFile = model;
            try
            {
                entry->tuner = std::make_unique<Tuner>(alpakaTune::makeTuner(
                    std::move(config),
                    makeFrameTuning(),
                    queue.getDevice(),
                    frameSpec.getExecutor(),
                    kernelIdentity));
            }
            catch(std::exception const& error)
            {
                throw std::runtime_error{
                    "alpakaTune initialization failed for "
                    + std::string{kernelIdentity}
                    + " (coverage=" + std::to_string(coverage)
                    + ", generated_frame_extents=" + std::to_string(extents.size())
                    + "): " + error.what()};
            }
        }

        auto const hostCallStarted = std::chrono::steady_clock::now();
        auto observed = [&]()
        {
            try
            {
                return entry->tuner->enqueueObserved(queue, frameSpec, bundle);
            }
            catch(std::exception const& error)
            {
                throw std::runtime_error{
                    "alpakaTune launch failed for "
                    + std::string{kernelIdentity}
                    + " (coverage=" + std::to_string(coverage)
                    + ", generated_frame_extents=" + std::to_string(extents.size())
                    + "): " + error.what()};
            }
        }();
        auto const hostCallSeconds
            = std::chrono::duration<double>{std::chrono::steady_clock::now() - hostCallStarted}.count();
        detail::appendMetrics(kernelIdentity, observed, hostCallSeconds);
        if(observed.learnedStatus && *observed.learnedStatus != "active")
            throw std::runtime_error{"learned_hybrid fallback is forbidden: " + *observed.learnedStatus};

        if(observed.measured || (observed.tuningComplete && !entry->completionTraced))
        {
            auto const candidatePosition = observed.configuration.empty() || extents.size() == 1u
                                               ? 0u
                                               : static_cast<std::size_t>(std::llround(
                                                     observed.configuration[0]
                                                     * static_cast<float>(extents.size() - 1u)));
            auto const selectedExtent = extents.at(candidatePosition);
            auto selectedFrames = selectedExtent;
            for(std::size_t dimension = 0u; dimension < selectedFrames.dim(); ++dimension)
                selectedFrames[dimension]
                    = frameSpec.getNumFrames()[dimension] * frameSpec.getFrameExtents()[dimension]
                      / selectedExtent[dimension];

            auto record = nlohmann::json::object();
            record["kernel"] = std::string{kernelIdentity};
            record["candidate_index"] = observed.candidateIndex;
            record["configuration"] = observed.configuration;
            record["runtime_seconds"] = observed.runtimeSeconds;
            record["recommendation_seconds"] = observed.recommendationSeconds;
            record["measured"] = observed.measured;
            record["tuning_complete"] = observed.tuningComplete;
            record["loaded_from_cache"] = observed.loadedFromCache;
            record["learned_status"] = observed.learnedStatus;
            record["learned_adapter_update_count"] = observed.learnedAdapterUpdateCount;
            record["elapsed_seconds"] = detail::campaignElapsedSeconds();
            record["original_num_frames"] = detail::vectorJson(frameSpec.getNumFrames());
            record["original_frame_extent"] = detail::vectorJson(frameSpec.getFrameExtents());
            record["selected_num_frames"] = detail::vectorJson(selectedFrames);
            record["selected_frame_extent"] = detail::vectorJson(selectedExtent);
            record["coverage"] = coverage;
            detail::appendTrace(record);
        }
        if(observed.tuningComplete)
            entry->completionTraced = true;
#else
        static_cast<void>(kernelIdentity);
        queue.enqueue(frameSpec, bundle);
#endif
        return TunedLaunchObservation{coverage};
    }
} // namespace hase::alpakaUtils
