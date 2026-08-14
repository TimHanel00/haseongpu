// Copyright 2026 Tim Hanel
//
// This file is part of HASEonGPU
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#if HASE_STATIC_FRAMESPEC_SELECTED
#    include <alpakaUtils/StaticFrameSpecWinners.hpp>
#endif

namespace hase::alpakaUtils
{
    struct FrameShape
    {
        std::uintmax_t numFrames;
        std::uintmax_t frameExtent;

        auto operator==(FrameShape const&) const -> bool = default;

        [[nodiscard]] constexpr auto workerCount() const -> std::uintmax_t
        {
            return numFrames * frameExtent;
        }
    };

    struct FrameSpecTuningSpace
    {
        std::vector<std::uintmax_t> numFrames;
        std::vector<std::uintmax_t> frameExtents;

        [[nodiscard]] auto candidateCount() const -> std::size_t
        {
            return numFrames.size() * frameExtents.size();
        }
    };

    /**
     * Build independent one-dimensional grid and frame-extent choices.
     *
     * The optional maximum extent comes from Alpaka's device-aware FrameSpec
     * selection. It widens explicitly narrow application launches without
     * exceeding Alpaka's selected device envelope. The two returned axes form
     * a Cartesian product: kernels using threadsInGrid with an IdxRange do not
     * require the physical worker count to equal the logical range size.
     */
    inline auto makeFrameSpecTuningSpace(
        FrameShape const original,
        std::uintmax_t const maximumFrameExtent = 0u) -> FrameSpecTuningSpace
    {
        auto result = FrameSpecTuningSpace{};
        if(original.numFrames == 0u || original.frameExtent == 0u)
            return result;

        auto const extentLimit = std::max(original.frameExtent, maximumFrameExtent);
        for(auto const extent : std::array<std::uintmax_t, 5u>{32u, 64u, 128u, 256u, 512u})
        {
            if(extent <= extentLimit)
                result.frameExtents.push_back(extent);
        }
        if(std::ranges::find(result.frameExtents, original.frameExtent) == result.frameExtents.end())
            result.frameExtents.push_back(original.frameExtent);

        if(original.numFrames > std::numeric_limits<std::uintmax_t>::max() / original.frameExtent)
            return {};
        auto const coverage = original.workerCount();
        for(auto const extent : result.frameExtents)
        {
            auto const frames = coverage / extent + static_cast<std::uintmax_t>(coverage % extent != 0u);
            if(std::ranges::find(result.numFrames, frames) == result.numFrames.end())
                result.numFrames.push_back(frames);
        }
        if(std::ranges::find(result.numFrames, original.numFrames) == result.numFrames.end())
            result.numFrames.push_back(original.numFrames);
        return result;
    }

    /** Generated winners used only by the ordinary, uninstrumented selected build. */
    inline auto staticSelectedFrameShape(std::string_view const kernel, FrameShape const original)
        -> std::optional<FrameShape>
    {
#if HASE_STATIC_FRAMESPEC_SELECTED
        for(auto const& winner : detail::generatedFrameSpecWinners)
        {
            if(winner.kernel == kernel && winner.originalNumFrames == original.numFrames
               && winner.originalFrameExtent == original.frameExtent)
                return FrameShape{winner.selectedNumFrames, winner.selectedFrameExtent};
        }
#else
        static_cast<void>(kernel);
        static_cast<void>(original);
#endif
        return std::nullopt;
    }

    template<typename T_FrameSpec>
    auto selectStaticFrameSpec(T_FrameSpec const& frameSpec, std::string_view const kernel) -> T_FrameSpec
    {
        auto numFrames = frameSpec.getNumFrames();
        auto frameExtents = frameSpec.getFrameExtents();
        auto const selected = staticSelectedFrameShape(
            kernel,
            FrameShape{static_cast<std::uintmax_t>(numFrames[0u]), static_cast<std::uintmax_t>(frameExtents[0u])});
        if(!selected)
        {
#if HASE_STATIC_FRAMESPEC_SELECTED
            if(
                kernel == "AccumulateForwardPhiAse" || kernel == "AccumulateForwardPhiAseReservoir"
                || kernel == "AccumulateReflectedForwardPhiAse" || kernel == "TraceGeneralPump")
                throw std::runtime_error{
                    "No generated static FrameSpec winner for " + std::string{kernel} + " with original shape "
                    + std::to_string(numFrames[0u]) + "x" + std::to_string(frameExtents[0u])};
#endif
            return frameSpec;
        }

        using FrameCountScalar = std::remove_cvref_t<decltype(numFrames[0u])>;
        using FrameExtentScalar = std::remove_cvref_t<decltype(frameExtents[0u])>;
        if(selected->numFrames > std::numeric_limits<FrameCountScalar>::max()
           || selected->frameExtent > std::numeric_limits<FrameExtentScalar>::max())
            return frameSpec;
        numFrames[0u] = static_cast<FrameCountScalar>(selected->numFrames);
        frameExtents[0u] = static_cast<FrameExtentScalar>(selected->frameExtent);
        return T_FrameSpec{numFrames, frameExtents, frameSpec.getExecutor()};
    }
} // namespace hase::alpakaUtils
