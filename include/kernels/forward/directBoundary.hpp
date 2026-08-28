#pragma once

#include <core/boundaryRayBuffer.hpp>
#include <kernels/forward/accumulation.hpp>

#include <cstdint>
#include <limits>

namespace hase::kernels::forward
{
    template<alpaka::concepts::SpecializationOf<hase::core::BoundaryRaySpans> T_Candidates>
    struct StoreDirectBoundary : ray::BoundaryPolicySrm<ray::srmPosition::Barycentric>
    {
        T_Candidates candidates;
        std::uint32_t candidateIndex;
        hase::data::AseDomainInterfaceView interfaceMap;
        std::uint32_t domain;
        std::uint32_t batch;
        std::uint32_t reflectionDepth;
        std::uint64_t historyId;
        bool useReflections;

        ALPAKA_FN_HOST_ACC constexpr StoreDirectBoundary(
            T_Candidates candidatesValue,
            std::uint32_t const candidateIndexValue,
            hase::data::AseDomainInterfaceView const interfaceMapValue,
            std::uint32_t const domainValue,
            std::uint32_t const batchValue,
            std::uint32_t const reflectionDepthValue,
            std::uint64_t const historyIdValue,
            bool const useReflectionsValue)
            : candidates{candidatesValue}
            , candidateIndex{candidateIndexValue}
            , interfaceMap{interfaceMapValue}
            , domain{domainValue}
            , batch{batchValue}
            , reflectionDepth{reflectionDepthValue}
            , historyId{historyIdValue}
            , useReflections{useReflectionsValue}
        {
        }

        [[nodiscard]] ALPAKA_FN_ACC bool isInteriorBoundary(
            hase::data::TraceView const&,
            unsigned const cell,
            unsigned const localFace) const
        {
            return interfaceMap.hasTarget(cell, localFace);
        }

        ALPAKA_FN_ACC ray::BoundaryResult operator()(
            alpaka::onAcc::concepts::Acc auto const&,
            hase::data::TraceView const& mesh,
            ray::State auto& rayState,
            unsigned const tet,
            unsigned const localFace)
        {
            auto const normal = outwardFaceNormal(mesh, tet, localFace);
            auto const faceIndex = tet * mesh.numberOfFacesPerCell + localFace;
            bool const isInterface = interfaceMap.hasTarget(tet, localFace);
            auto const interaction = boundaryInteraction(
                rayState.direction,
                normal,
                isInterface ? static_cast<double>(interfaceMap.sourceRefractiveIndices[faceIndex])
                            : static_cast<double>(mesh.getSurfaceRefractiveIndexInside(tet, localFace)),
                isInterface ? static_cast<double>(interfaceMap.targetRefractiveIndices[faceIndex])
                            : static_cast<double>(mesh.getSurfaceRefractiveIndexOutside(tet, localFace)),
                isInterface ? static_cast<double>(interfaceMap.reflectivities[faceIndex])
                            : static_cast<double>(mesh.getSurfaceReflectivity(tet, localFace)));
            auto const barycentric = ray::triangleBarycentricCoordinates(mesh, tet, localFace, rayState.position);
            auto const store = [&](hase::core::Direction const direction,
                                   double const weight,
                                   std::uint32_t const targetDomain,
                                   std::uint32_t const targetCell,
                                   std::uint32_t const targetFace,
                                   std::uint32_t const branch,
                                   std::uint64_t const branchHistory)
            {
                auto const index = 2u * candidateIndex + branch;
                candidates.positions.x[index] = rayState.position.x;
                candidates.positions.y[index] = rayState.position.y;
                candidates.positions.z[index] = rayState.position.z;
                candidates.directions.x[index] = direction.x;
                candidates.directions.y[index] = direction.y;
                candidates.directions.z[index] = direction.z;
                candidates.faceBarycentric.x[index] = barycentric[0u];
                candidates.faceBarycentric.y[index] = barycentric[1u];
                candidates.faceBarycentric.z[index] = barycentric[2u];
                candidates.weights[index] = weight;
                candidates.wavelengths[index] = rayState.wavelength;
                candidates.targetDomains[index] = targetDomain;
                candidates.targetCells[index] = targetCell;
                candidates.targetFaces[index] = targetFace;
                candidates.batches[index] = batch;
                candidates.reflectionDepths[index] = reflectionDepth;
                candidates.historyIds[index] = branchHistory;
            };
            double const boundaryWeight = rayState.weight * rayState.accumulatedGain;
            bool const hasTransmission
                = interfaceMap.hasTarget(tet, localFace) && !interaction.totalInternalReflection;
            auto const weights = splitBoundaryWeights(boundaryWeight, interaction, hasTransmission, useReflections);
            store(interaction.reflected, weights.reflected, domain, tet, localFace, 0u, 2u * historyId);
            store(
                interaction.transmitted,
                weights.transmitted,
                hasTransmission ? interfaceMap.targetDomains[faceIndex] : hase::data::invalidDomainId,
                hasTransmission ? interfaceMap.targetCells[faceIndex] : std::numeric_limits<std::uint32_t>::max(),
                hasTransmission ? interfaceMap.targetFaces[faceIndex] : std::numeric_limits<std::uint32_t>::max(),
                1u,
                2u * historyId + 1u);
            return ray::BoundaryResult::stop;
        }
    };

    struct AccumulateForwardPhiAseDirect
    {
        ALPAKA_FN_HOST_ACC void walk(
            alpaka::onAcc::concepts::Acc auto const& acc,
            concepts::TracePolicyList auto const tracePolicies,
            hase::data::TraceView const& mesh,
            unsigned const tet,
            hase::core::Position const origin,
            hase::core::Direction const direction,
            int const forbiddenFace,
            double const weight,
            double const wavelength,
            unsigned const batch,
            alpaka::concepts::SpecializationOf<ForwardAccumulationSpans> auto accumulation,
            alpaka::concepts::SpecializationOf<hase::core::BoundaryRaySpans> auto candidates,
            std::uint32_t const candidateIndex,
            hase::data::AseDomainInterfaceView const interfaceMap,
            std::uint32_t const domain,
            std::uint32_t const reflectionDepth,
            std::uint64_t const historyId,
            bool const useReflections) const
        {
            ForwardAseRayState rayState;
            rayState.position = origin;
            rayState.direction = direction;
            rayState.cell = tet;
            rayState.forbiddenFace = forbiddenFace;
            rayState.weight = weight;
            rayState.wavelength = wavelength;
            rayState.rseBatch = batch;
            auto const cellBehaviour = MakeForwardAseCellPolicy{}(tracePolicies.getDiagnostics(), accumulation);
            auto const failureBehaviour
                = MakeForwardRayFailureBehaviour{}(tracePolicies.getDiagnostics(), accumulation);
            ray::walk(
                acc,
                mesh,
                rayState,
                ray::RayWalkBehaviour{
                    cellBehaviour,
                    StoreDirectBoundary<ALPAKA_TYPEOF(candidates)>{
                        candidates,
                        candidateIndex,
                        interfaceMap,
                        domain,
                        batch,
                        reflectionDepth,
                        historyId,
                        useReflections},
                    failureBehaviour});
        }

        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            concepts::TracePolicyList auto const tracePolicies,
            hase::data::TraceView const mesh,
            unsigned const forwardRayCount,
            unsigned const batch,
            double const sourceStrengthTotal,
            alpaka::concepts::SpecializationOf<ForwardAccumulationSpans> auto accumulation,
            alpaka::concepts::SpecializationOf<hase::core::BoundaryRaySpans> auto candidates,
            unsigned const rngSeed,
            hase::data::AseDomainInterfaceView const interfaceMap,
            bool const useReflections) const
        {
            for(auto [rayNumber] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{forwardRayCount}))
            {
                unsigned const batchSeed = rseBatchSeed(rngSeed, batch);
                auto rng = alpaka::rand::engine::Philox4x32x10{batchSeed, rayHistoryId(0u, rayNumber)};
                unsigned const tet = sampleStratifiedVolumeBySourceStrength(
                    mesh,
                    sourceStrengthTotal,
                    rayNumber,
                    forwardRayCount,
                    rseBatchSourceStratificationOffset(rngSeed, batch),
                    rng);
                auto const material = mesh.getMaterialId(tet);
                auto const spectrumSize = mesh.crossSectionCount(material);
                auto const spectrumIndex = stratifiedSpectrumIndex(
                    spectrumSize,
                    rayNumber,
                    forwardRayCount,
                    rseBatchSpectrumStratificationPhase(rngSeed, batch, spectrumSize));
                auto const origin = samplePointInVolume(mesh, tet, rng);
                auto const direction = sampleIsotropicDirection(rng);
                walk(
                    acc,
                    tracePolicies,
                    mesh,
                    tet,
                    origin,
                    direction,
                    -1,
                    sourceStrengthTotal > 0.0 ? 1.0 : 0.0,
                    spectrumSize == 0u ? 0.0 : mesh.emissionWavelength(material, spectrumIndex),
                    batch,
                    accumulation,
                    candidates,
                    rayNumber,
                    interfaceMap,
                    interfaceMap.sourceDomain(tet),
                    1u,
                    rayHistoryId(0u, rayNumber),
                    useReflections);
            }
        }
    };

    /** @brief Launch exactly one configured domain population from its precomputed source CDF. */
    struct AccumulateDomainForwardPhiAseDirect
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            concepts::TracePolicyList auto const tracePolicies,
            hase::data::TraceView const mesh,
            hase::data::AseDomainSourceView const sources,
            std::uint32_t const domain,
            unsigned const domainRayCount,
            unsigned const candidateOffset,
            unsigned const batch,
            double const sourceWeight,
            alpaka::concepts::SpecializationOf<ForwardAccumulationSpans> auto accumulation,
            alpaka::concepts::SpecializationOf<hase::core::BoundaryRaySpans> auto candidates,
            unsigned const rngSeed,
            hase::data::AseDomainInterfaceView const interfaceMap,
            bool const useReflections) const
        {
            AccumulateForwardPhiAseDirect walker;
            auto const begin = sources.offsets[domain];
            auto const end = sources.offsets[domain + 1u];
            double const total = sources.sourceStrengthTotals[domain];
            double const prefixBase = begin == 0u ? 0.0 : sources.sourceStrengthPrefix[begin - 1u];
            for(auto [rayNumber] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{domainRayCount}))
            {
                auto rng = alpaka::rand::engine::Philox4x32x10{
                    rseBatchSeed(rngSeed, batch),
                    rayHistoryId(0u, candidateOffset + rayNumber)};
                double const target = stratifiedUnitInterval(
                                          rayNumber,
                                          domainRayCount,
                                          rseBatchSourceStratificationOffset(rngSeed, batch))
                                          * total
                                      + prefixBase;
                std::uint32_t lower = begin;
                std::uint32_t upper = end;
                while(lower < upper)
                {
                    auto const middle = lower + (upper - lower) / 2u;
                    if(sources.sourceStrengthPrefix[middle] <= target)
                        lower = middle + 1u;
                    else
                        upper = middle;
                }
                auto const sourceIndex = lower < end ? lower : end - 1u;
                auto const tet = sources.globalCells[sourceIndex];
                auto const material = mesh.getMaterialId(tet);
                auto const spectrumSize = mesh.crossSectionCount(material);
                auto const spectrumIndex = stratifiedSpectrumIndex(
                    spectrumSize,
                    rayNumber,
                    domainRayCount,
                    rseBatchSpectrumStratificationPhase(rngSeed, batch, spectrumSize));
                auto const origin = samplePointInVolume(mesh, tet, rng);
                auto const direction = sampleIsotropicDirection(rng);
                walker.walk(
                    acc,
                    tracePolicies,
                    mesh,
                    tet,
                    origin,
                    direction,
                    -1,
                    sourceWeight,
                    spectrumSize == 0u ? 0.0 : mesh.emissionWavelength(material, spectrumIndex),
                    batch,
                    accumulation,
                    candidates,
                    candidateOffset + rayNumber,
                    interfaceMap,
                    domain,
                    1u,
                    rayHistoryId(0u, candidateOffset + rayNumber),
                    useReflections);
            }
        }
    };

    struct AccumulateParticleCombedForwardPhiAse
    {
        ALPAKA_FN_ACC void operator()(
            alpaka::onAcc::concepts::Acc auto const& acc,
            concepts::TracePolicyList auto const tracePolicies,
            hase::data::TraceView const mesh,
            unsigned const rayCount,
            alpaka::concepts::SpecializationOf<ForwardAccumulationSpans> auto accumulation,
            alpaka::concepts::SpecializationOf<hase::core::BoundaryRaySpans> auto input,
            alpaka::concepts::IView<std::uint32_t> auto const selected,
            alpaka::concepts::IView<double> auto const selectedWeights,
            alpaka::concepts::SpecializationOf<hase::core::BoundaryRaySpans> auto output,
            hase::data::AseDomainInterfaceView const interfaceMap,
            std::uint32_t const reflectionDepth,
            bool const useReflections) const
        {
            AccumulateForwardPhiAseDirect walker;
            for(auto [rayNumber] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{rayCount}))
            {
                if(selectedWeights[rayNumber] <= 0.0)
                    continue;
                auto const candidate = selected[rayNumber];
                auto const origin = input.positions.at(candidate);
                auto const direction = normalize(input.directions.at(candidate));
                walker.walk(
                    acc,
                    tracePolicies,
                    mesh,
                    input.targetCells[candidate],
                    origin,
                    direction,
                    static_cast<int>(input.targetFaces[candidate]),
                    selectedWeights[rayNumber],
                    input.wavelengths[candidate],
                    input.batches[candidate],
                    accumulation,
                    output,
                    rayNumber,
                    interfaceMap,
                    input.targetDomains[candidate],
                    input.reflectionDepths[candidate] + 1u,
                    input.historyIds[candidate] ^ ((static_cast<std::uint64_t>(reflectionDepth) << 32u) | rayNumber),
                    useReflections);
            }
        }
    };
} // namespace hase::kernels::forward
