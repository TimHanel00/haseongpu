// Copyright 2026 Tim Hanel
//
// This file is part of HASEonGPU
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <alpakaUtils/FrameSpecPolicy.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace
{
    struct TestFrameSpec
    {
        std::array<unsigned, 1u> numFrames;
        std::array<unsigned, 1u> frameExtents;
        unsigned executor;

        auto getNumFrames() const -> std::array<unsigned, 1u> const&
        {
            return numFrames;
        }

        auto getFrameExtents() const -> std::array<unsigned, 1u> const&
        {
            return frameExtents;
        }

        auto getExecutor() const -> unsigned
        {
            return executor;
        }
    };
} // namespace

TEST_CASE("FrameSpec tuning axes form a materially expanded Cartesian space")
{
    auto const original = hase::alpakaUtils::FrameShape{97u, 512u};
    auto const space = hase::alpakaUtils::makeFrameSpecTuningSpace(original);

    REQUIRE(space.frameExtents == std::vector<std::uintmax_t>{32u, 64u, 128u, 256u, 512u});
    REQUIRE(space.numFrames == std::vector<std::uintmax_t>{1552u, 776u, 388u, 194u, 97u});
    REQUIRE(space.candidateCount() == 25u);

    auto coverages = std::vector<std::uintmax_t>{};
    for(auto const frames : space.numFrames)
        for(auto const extent : space.frameExtents)
            coverages.push_back(frames * extent);
    REQUIRE(std::ranges::any_of(coverages, [&](auto const value) { return value != original.workerCount(); }));
}

TEST_CASE("FrameSpec candidates retain the device-selected extent legality envelope")
{
    auto const space = hase::alpakaUtils::makeFrameSpecTuningSpace({3u, 96u});

    REQUIRE_FALSE(space.frameExtents.empty());
    REQUIRE(std::ranges::all_of(space.frameExtents, [](auto const extent) { return extent > 0u && extent <= 96u; }));
    REQUIRE(std::ranges::find(space.frameExtents, 96u) != space.frameExtents.end());
    REQUIRE(std::ranges::find(space.numFrames, 3u) != space.numFrames.end());
}

TEST_CASE("Explicit narrow launches can use a wider device-selected tuning envelope")
{
    auto const space = hase::alpakaUtils::makeFrameSpecTuningSpace({977u, 128u}, 512u);

    REQUIRE(space.frameExtents == std::vector<std::uintmax_t>{32u, 64u, 128u, 256u, 512u});
    REQUIRE(space.numFrames == std::vector<std::uintmax_t>{3908u, 1954u, 977u, 489u, 245u});
    REQUIRE(space.candidateCount() == 25u);
}

TEST_CASE("Independent worker counts cover a grid-stride logical range exactly once")
{
    constexpr auto logicalRayCount = std::uintmax_t{1003u};
    for(auto const workerCount : std::array<std::uintmax_t, 4u>{32u, 288u, 1024u, 4096u})
    {
        auto visits = std::vector<unsigned>(logicalRayCount, 0u);
        for(auto worker = std::uintmax_t{0u}; worker < workerCount; ++worker)
            for(auto ray = worker; ray < logicalRayCount; ray += workerCount)
                ++visits[ray];
        REQUIRE(std::ranges::all_of(visits, [](auto const count) { return count == 1u; }));
    }
}

TEST_CASE("Ordinary selected launch routing uses generated per-context winners")
{
    using hase::alpakaUtils::FrameShape;
    using hase::alpakaUtils::staticSelectedFrameShape;

#if HASE_STATIC_FRAMESPEC_SELECTED
    REQUIRE((staticSelectedFrameShape("AccumulateForwardPhiAse", {52u, 512u}) == FrameShape{201u, 128u}));
    REQUIRE(
        (staticSelectedFrameShape("AccumulateForwardPhiAseReservoir", {977u, 128u})
         == FrameShape{3908u, 32u}));
    REQUIRE(
        (staticSelectedFrameShape("AccumulateReflectedForwardPhiAse", {977u, 128u})
         == FrameShape{1954u, 64u}));
    REQUIRE((staticSelectedFrameShape("TraceGeneralPump", {97u, 512u}) == FrameShape{123u, 256u}));
#else
    REQUIRE_FALSE(staticSelectedFrameShape("AccumulateForwardPhiAse", {52u, 512u}));
    REQUIRE_FALSE(staticSelectedFrameShape("AccumulateForwardPhiAseReservoir", {977u, 128u}));
    REQUIRE_FALSE(staticSelectedFrameShape("AccumulateReflectedForwardPhiAse", {977u, 128u}));
    REQUIRE_FALSE(staticSelectedFrameShape("TraceGeneralPump", {97u, 512u}));
#endif
    REQUIRE_FALSE(staticSelectedFrameShape("TraceGeneralPump", {96u, 512u}));
    REQUIRE_FALSE(staticSelectedFrameShape("UnselectedKernel", {97u, 512u}));
#if HASE_STATIC_FRAMESPEC_SELECTED
    REQUIRE_THROWS(hase::alpakaUtils::selectStaticFrameSpec(TestFrameSpec{{96u}, {512u}, 7u}, "TraceGeneralPump"));
#endif

    auto const selected = hase::alpakaUtils::selectStaticFrameSpec(
        TestFrameSpec{{97u}, {512u}, 7u},
        "TraceGeneralPump");
#if HASE_STATIC_FRAMESPEC_SELECTED
    REQUIRE((selected.numFrames == std::array<unsigned, 1u>{123u}));
    REQUIRE((selected.frameExtents == std::array<unsigned, 1u>{256u}));
#else
    REQUIRE((selected.numFrames == std::array<unsigned, 1u>{97u}));
    REQUIRE((selected.frameExtents == std::array<unsigned, 1u>{512u}));
#endif
    REQUIRE(selected.executor == 7u);

    auto const unchanged
        = hase::alpakaUtils::selectStaticFrameSpec(TestFrameSpec{{11u}, {64u}, 9u}, "OtherKernel");
    REQUIRE((unchanged.numFrames == std::array<unsigned, 1u>{11u}));
    REQUIRE((unchanged.frameExtents == std::array<unsigned, 1u>{64u}));
    REQUIRE(unchanged.executor == 9u);
}
