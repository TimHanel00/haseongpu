#pragma once

#include <data/TraceData.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace hase::data
{
    using DomainId = std::uint32_t;
    using BatchId = std::uint32_t;
    inline constexpr DomainId invalidDomainId = std::numeric_limits<DomainId>::max();

    /** @brief Device-safe per-face routing metadata for one domain-local trace. */
    struct AseDomainInterfaceView
    {
        std::span<DomainId const> targetDomains;
        std::span<std::uint32_t const> targetCells;
        std::span<std::uint32_t const> targetFaces;
        std::span<DomainId const> cellDomains;
        std::span<float const> reflectivities;
        std::span<float const> sourceRefractiveIndices;
        std::span<float const> targetRefractiveIndices;

        [[nodiscard]] ALPAKA_FN_ACC bool hasTarget(unsigned const cell, unsigned const face) const
        {
            auto const index = cell * tet4FaceCount + face;
            return index < targetDomains.size() && targetDomains[index] != invalidDomainId;
        }

        [[nodiscard]] ALPAKA_FN_ACC DomainId sourceDomain(unsigned const cell) const
        {
            return cell < cellDomains.size() ? cellDomains[cell] : DomainId{0u};
        }
    };

    /** @brief Flattened domain-cell sampling tables retained on the tracing device. */
    struct AseDomainSourceView
    {
        std::span<std::uint32_t const> offsets;
        std::span<std::uint32_t const> globalCells;
        std::span<double const> sourceStrengthPrefix;
        std::span<double const> sourceStrengthTotals;
    };

    /** @brief One directed connection between component-local trace faces. */
    struct AseDomainInterface
    {
        DomainId sourceDomain{};
        DomainId targetDomain{};
        std::uint32_t sourceCell{};
        std::uint32_t sourceFace{};
        std::uint32_t targetCell{};
        std::uint32_t targetFace{};
        std::uint32_t boundaryId{};
        double reflectivity{};
        double sourceRefractiveIndex{1.0};
        double targetRefractiveIndex{1.0};
    };

    /** @brief Prepared device-tracing unit corresponding to one OpticalComponent. */
    struct AseDomain
    {
        DomainId id{};
        TraceData trace;
        std::vector<std::uint32_t> localToGlobalCells;
        std::vector<std::uint32_t> localToGlobalPoints;
        std::vector<DomainId> cellDomains;
        std::vector<DomainId> boundaryTargetDomains;
        std::vector<std::uint32_t> boundaryTargetCells;
        std::vector<std::uint32_t> boundaryTargetFaces;
        std::vector<float> boundaryReflectivities;
        std::vector<float> boundarySourceRefractiveIndices;
        std::vector<float> boundaryTargetRefractiveIndices;
        std::optional<std::uint64_t> requestedRays;
    };

    /** @brief Component-local traces and their compact directed interface graph. */
    struct AseDomainGraph
    {
        std::vector<AseDomain> domains;
        std::vector<AseDomainInterface> interfaces;
        std::vector<DomainId> globalCellDomains;
        std::vector<DomainId> globalBoundaryTargetDomains;
        std::vector<std::uint32_t> globalBoundaryTargetCells;
        std::vector<std::uint32_t> globalBoundaryTargetFaces;
        std::vector<float> globalBoundaryReflectivities;
        std::vector<float> globalBoundarySourceRefractiveIndices;
        std::vector<float> globalBoundaryTargetRefractiveIndices;
        std::vector<std::uint32_t> domainCellOffsets;
        std::vector<std::uint32_t> domainGlobalCells;
        std::vector<double> domainSourceStrengthPrefix;
        std::vector<double> domainSourceStrengthTotals;
    };
} // namespace hase::data
