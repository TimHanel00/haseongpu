// Copyright 2026 Tim Hanel
//
// This file is part of HASEonGPU
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <alpaka/alpaka.hpp>

#include <string_view>

#if HASE_ENABLE_ALPAKATUNE
#    include <alpakaTune/alpakaTune.hpp>
#    include <nlohmann/json.hpp>

#    include <algorithm>
#    include <chrono>
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
    } // namespace detail
#endif

    /**
     * Measure one natural kernel launch without changing its FrameSpec.
     *
     * The tuning bundle contains exactly the application's original frame
     * count and extent. With an online-adaptive YAML configuration that has no
     * queue and no horizon, every call is therefore a direct device-event
     * measurement of the unchanged launch rather than an active search.
     */
    template<typename T_Queue, typename T_FrameSpec, typename T_Kernel, typename... T_Args>
    void timedEnqueue(
        T_Queue const& queue,
        T_FrameSpec const& frameSpec,
        alpaka::KernelBundle<T_Kernel, T_Args...> const& bundle,
        std::string_view kernelIdentity)
    {
#if HASE_ENABLE_ALPAKATUNE
        auto makeFrameTuning = [&]()
        {
            return alpakaTune::constrain(
                alpakaTune::TunableBundle{
                    alpakaTune::tuneFrameExtent(
                        frameSpec,
                        alpakaTune::RVals<ALPAKA_TYPEOF(frameSpec.getFrameExtents())>{
                            std::vector{frameSpec.getFrameExtents()}}),
                    alpakaTune::tuneNumFrames(
                        frameSpec,
                        alpakaTune::RVals<ALPAKA_TYPEOF(frameSpec.getNumFrames())>{
                            std::vector{frameSpec.getNumFrames()}})},
                alpakaTune::defaultFrameExtentShape(frameSpec));
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
            explicit TunerEntry(Device value) : device{std::move(value)}
            {
            }

            Device device;
            std::unique_ptr<Tuner> tuner;
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
        if(!entry->tuner)
        {
            auto config = alpakaTune::TunerConfig::fromYaml(detail::requiredTimingEnvironment("ALPAKA_TUNE_CONFIG"));
            if(auto const path = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_HISTORY"); !path.empty())
                config.history.file = path;
            config.completeHistory.file = detail::requiredTimingEnvironment("HASE_ALPAKATUNE_COMPLETE_HISTORY");
            entry->tuner = std::make_unique<Tuner>(alpakaTune::makeTuner(
                std::move(config),
                makeFrameTuning(),
                queue.getDevice(),
                frameSpec.getExecutor(),
                kernelIdentity));
        }

        auto const observation = entry->tuner->enqueueObserved(queue, frameSpec, bundle);
        if(auto const tracePath = detail::optionalTimingEnvironment("HASE_ALPAKATUNE_TRACE"); !tracePath.empty())
        {
            auto const info = entry->tuner->info();
            auto record = nlohmann::json::object();
            record["kernel"] = kernelIdentity;
            record["candidate_index"] = observation.candidateIndex;
            record["runtime_seconds"] = observation.runtimeSeconds;
            record["runtime_measurement_source"]
                = alpakaTune::runtimeMeasurementSourceName(observation.runtimeMeasurementSource);
            record["measured"] = observation.measured;
            record["elapsed_seconds"] = detail::timingCampaignElapsedSeconds();
            record["num_frames"] = detail::timingVectorJson(frameSpec.getNumFrames());
            record["frame_extent"] = detail::timingVectorJson(frameSpec.getFrameExtents());
            record["candidate_count"] = info.candidateCount;
            record["active_queue"] = false;
            detail::timingTraceStore().stage(tracePath, record);
        }
#else
        static_cast<void>(kernelIdentity);
        queue.enqueue(frameSpec, bundle);
#endif
    }
} // namespace hase::alpakaUtils
