#pragma once

#include <core/Runtime.hpp>
#include <core/domainSchedule.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hase::core
{
    /** Every active statistical batch must be able to sample every emitting domain. */
    [[nodiscard]] inline std::uint32_t domainRseBatchCount(
        std::span<DomainQuota const> const quotas,
        std::uint32_t requested)
    {
        for(auto const& quota : quotas)
            if(quota.sourceStrength > 0.0)
                requested = static_cast<std::uint32_t>(std::min<std::uint64_t>(requested, quota.rayCount));
        if(requested == 0u)
            throw std::invalid_argument("positive ASE sources require at least one statistical batch");
        return requested;
    }

    [[nodiscard]] inline std::uint32_t domainBatchRayCount(
        std::uint32_t const domainCount,
        std::uint32_t const batch,
        std::uint32_t const batchCount)
    {
        return domainCount / batchCount + (batch < domainCount % batchCount ? 1u : 0u);
    }

    /** Importance weight making each batch estimate the complete physical source. */
    [[nodiscard]] inline double domainBatchSourceWeight(
        double const domainStrength,
        double const sourceStrength,
        std::uint32_t const domainRays,
        std::uint32_t const batchRays)
    {
        if(domainStrength == 0.0 || sourceStrength == 0.0)
            return 0.0;
        if(domainRays == 0u)
            throw std::invalid_argument("statistical batch omitted a positive ASE source");
        return (domainStrength / sourceStrength) * (static_cast<double>(batchRays) / domainRays);
    }

    struct ForwardLaunchPlan
    {
        std::uint32_t target;
        std::vector<std::uint32_t> domainCounts;
    };

    /**
     * Coalesce adaptive increments when a small quota cannot represent its source
     * in every batch. Retain enough histories for every later complete-source launch.
     * The final budget and each domain's exact quota remain unchanged.
     */
    [[nodiscard]] inline ForwardLaunchPlan planForwardLaunch(
        AseTraceControls const& experiment,
        ExecutionPolicy const& compute,
        std::span<DomainQuota const> const quotas,
        std::span<std::uint64_t const> const completed,
        std::uint32_t const batchCount,
        std::uint32_t const previousTarget,
        std::uint32_t& increase)
    {
        for(;; ++increase)
        {
            auto const target = adaptiveRayTarget(experiment, compute, increase);
            if(target <= previousTarget)
            {
                if(increase < compute.adaptiveSteps && compute.adaptiveSteps != 0u)
                    continue;
                throw std::runtime_error("adaptive ASE sampling cannot make progress");
            }
            auto counts = allocateDomainLaunchCounts(quotas, completed, target - previousTarget);
            bool const final
                = experiment.forwardRayCount != 0u || compute.adaptiveSteps == 0u || target == experiment.maxRays;
            bool complete = true;
            for(std::size_t domain = 0u; domain < quotas.size(); ++domain)
                if(quotas[domain].sourceStrength > 0.0)
                    complete
                        = complete && counts[domain] >= batchCount
                          && (final || quotas[domain].rayCount - completed[domain] - counts[domain] >= batchCount);
            if(complete)
                return {target, std::move(counts)};
            if(final)
                throw std::runtime_error("ASE quotas cannot represent every source in each statistical batch");
        }
    }
} // namespace hase::core
