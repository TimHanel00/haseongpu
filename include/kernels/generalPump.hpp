/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/memory.hpp>
#include <alpakaUtils/utils.hpp>
#include <core/mesh.hpp>
#include <core/simulationRunControl.hpp>
#include <kernels/forward/rayTransition.hpp>
#include <kernels/forward/rayWalk.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hase::kernels
{
    struct PumpBoundaryFace
    {
        unsigned cell = 0u;
        unsigned localFace = 0u;
        int domain = 0;
        std::array<hase::core::Point, 3u> vertices;
        hase::core::Point centroid;
        hase::core::Point normal;
        double area = 0.0;
    };

    struct PumpRayBatch
    {
        std::vector<double> originX, originY, originZ;
        std::vector<double> directionX, directionY, directionZ;
        std::vector<double> power, wavelength, sigmaAbsorption, sigmaEmission;
        std::vector<unsigned> cell;
        std::vector<int> forbiddenFace, exitFace;

        [[nodiscard]] std::size_t size() const
        {
            return power.size();
        }
    };

    [[nodiscard]] inline bool containsDomain(std::vector<int> const& domains, int const domain)
    {
        return std::ranges::find(domains, domain) != domains.end();
    }

    [[nodiscard]] inline hase::core::Point hostPoint(hase::core::HostMesh const& mesh, unsigned const point)
    {
        return {
            mesh.points[point],
            mesh.points[point + mesh.numberOfMeshPoints],
            mesh.points[point + 2u * mesh.numberOfMeshPoints]};
    }

    [[nodiscard]] inline std::vector<PumpBoundaryFace> pumpBoundaryFaces(
        hase::core::HostMesh const& mesh,
        std::vector<int> const& domains)
    {
        std::vector<PumpBoundaryFace> result;
        for(unsigned cell = 0u; cell < mesh.numberOfCells; ++cell)
        {
            for(unsigned face = 0u; face < mesh.numberOfFacesPerCell; ++face)
            {
                unsigned const faceIndex = cell * mesh.numberOfFacesPerCell + face;
                int const domain = mesh.cellFaceBoundaries[faceIndex];
                if(mesh.cellNeighborCells[faceIndex] >= 0 || !containsDomain(domains, domain))
                    continue;
                PumpBoundaryFace info;
                info.cell = cell;
                info.localFace = face;
                info.domain = domain;
                for(unsigned vertex = 0u; vertex < 3u; ++vertex)
                {
                    int const point = mesh.cellFaces[faceIndex * 3u + vertex];
                    if(point < 0)
                        throw std::runtime_error("pump boundary face contains an invalid point");
                    info.vertices[vertex] = hostPoint(mesh, static_cast<unsigned>(point));
                }
                info.centroid = (info.vertices[0] + info.vertices[1] + info.vertices[2]) * (1.0 / 3.0);
                auto normal
                    = hase::core::cross(info.vertices[1] - info.vertices[0], info.vertices[2] - info.vertices[0]);
                double const twiceArea = normal.euclidLength();
                if(twiceArea <= 0.0)
                    continue;
                info.area = 0.5 * twiceArea;
                info.normal = normal * (1.0 / twiceArea);
                hase::core::Point const center{
                    mesh.cellCenters[cell],
                    mesh.cellCenters[cell + mesh.numberOfCells],
                    mesh.cellCenters[cell + 2u * mesh.numberOfCells]};
                if(hase::core::dot(info.normal, center - info.centroid) > 0.0)
                    info.normal = info.normal * -1.0;
                result.push_back(info);
            }
        }
        return result;
    }

    [[nodiscard]] inline hase::core::Point hostNormalize(hase::core::Point const value)
    {
        double const length = value.euclidLength();
        if(length <= 0.0)
            return {0.0, 0.0, 0.0};
        return value * (1.0 / length);
    }

    [[nodiscard]] inline hase::core::Point perpendicular(hase::core::Point const normal)
    {
        hase::core::Point reference
            = std::abs(normal.x) < 0.9 ? hase::core::Point{1.0, 0.0, 0.0} : hase::core::Point{0.0, 1.0, 0.0};
        return hostNormalize(hase::core::cross(normal, reference));
    }

    [[nodiscard]] inline double pumpProfileWeight(
        hase::core::PumpProfileParameters const& profile,
        hase::core::Point const point)
    {
        if(profile.kind == 0u)
            return 1.0;
        hase::core::Point const relative
            = point - hase::core::Point{profile.center[0], profile.center[1], profile.center[2]};
        double const u
            = hase::core::dot(relative, hase::core::Point{profile.axisU[0], profile.axisU[1], profile.axisU[2]})
              / profile.radiusU;
        double const v
            = hase::core::dot(relative, hase::core::Point{profile.axisV[0], profile.axisV[1], profile.axisV[2]})
              / profile.radiusV;
        return std::exp(-std::pow(std::sqrt(u * u + v * v), profile.exponent));
    }

    template<typename T_Rng>
    [[nodiscard]] inline hase::core::Point sampleTriangle(PumpBoundaryFace const& face, T_Rng& rng)
    {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double u = uniform(rng);
        double v = uniform(rng);
        if(u + v > 1.0)
        {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        return face.vertices[0] + (face.vertices[1] - face.vertices[0]) * u
               + (face.vertices[2] - face.vertices[0]) * v;
    }

    [[nodiscard]] inline PumpRayBatch samplePumpSource(
        hase::core::HostMesh const& mesh,
        hase::core::PumpSourceParameters const& source,
        unsigned const rayCount,
        std::uint32_t const seed)
    {
        auto const faces = pumpBoundaryFaces(mesh, source.surfaces);
        if(faces.empty())
            throw std::runtime_error("pump source selected no exterior boundary faces");
        std::vector<double> areas;
        areas.reserve(faces.size());
        for(auto const& face : faces)
            areas.push_back(face.area);
        std::discrete_distribution<std::size_t> faceDistribution(areas.begin(), areas.end());
        std::discrete_distribution<std::size_t> spectrumDistribution(
            source.spectralWeights.begin(),
            source.spectralWeights.end());
        std::discrete_distribution<std::size_t> angularDistribution(
            source.angularWeights.begin(),
            source.angularWeights.end());
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        PumpRayBatch batch;
        auto reserve = [rayCount](auto& values) { values.reserve(rayCount); };
        reserve(batch.originX);
        reserve(batch.originY);
        reserve(batch.originZ);
        reserve(batch.directionX);
        reserve(batch.directionY);
        reserve(batch.directionZ);
        reserve(batch.power);
        reserve(batch.wavelength);
        reserve(batch.sigmaAbsorption);
        reserve(batch.sigmaEmission);
        reserve(batch.cell);
        reserve(batch.forbiddenFace);
        reserve(batch.exitFace);
        for(unsigned ray = 0u; ray < rayCount; ++ray)
        {
            PumpBoundaryFace const* face = nullptr;
            hase::core::Point origin;
            bool accepted = false;
            for(unsigned attempt = 0u; attempt < 100000u; ++attempt)
            {
                face = &faces[faceDistribution(rng)];
                origin = sampleTriangle(*face, rng);
                if(uniform(rng) <= pumpProfileWeight(source.profile, origin))
                {
                    accepted = true;
                    break;
                }
            }
            if(!accepted)
                throw std::runtime_error("pump spatial profile rejection sampling did not converge");

            std::size_t const angular = angularDistribution(rng);
            double const theta = source.polarAngles[angular];
            double const phi = source.azimuthalAngles[angular];
            hase::core::Point const inward = face->normal * -1.0;
            hase::core::Point const u = perpendicular(inward);
            hase::core::Point const v = hase::core::cross(inward, u);
            hase::core::Point const direction = hostNormalize(
                inward * std::cos(theta) + u * (std::sin(theta) * std::cos(phi))
                + v * (std::sin(theta) * std::sin(phi)));
            std::size_t const spectrum = spectrumDistribution(rng);

            batch.originX.push_back(origin.x);
            batch.originY.push_back(origin.y);
            batch.originZ.push_back(origin.z);
            batch.directionX.push_back(direction.x);
            batch.directionY.push_back(direction.y);
            batch.directionZ.push_back(direction.z);
            batch.power.push_back(source.totalPower / static_cast<double>(rayCount));
            batch.wavelength.push_back(source.wavelengths[spectrum]);
            batch.sigmaAbsorption.push_back(source.sigmaAbsorption[spectrum]);
            batch.sigmaEmission.push_back(source.sigmaEmission[spectrum]);
            batch.cell.push_back(face->cell);
            batch.forbiddenFace.push_back(static_cast<int>(face->localFace));
            batch.exitFace.push_back(-1);
        }
        return batch;
    }

    struct TraceGeneralPump
    {
        double planckConstant = 6.62607015e-34;
        double speedOfLight = 299792458.0;

        template<
            typename T_Acc,
            typename T_BetaVolumeView,
            typename T_OriginXView,
            typename T_OriginYView,
            typename T_OriginZView,
            typename T_DirectionXView,
            typename T_DirectionYView,
            typename T_DirectionZView,
            typename T_PowerView,
            typename T_WavelengthView,
            typename T_SigmaAbsorptionView,
            typename T_SigmaEmissionView,
            typename T_CellView,
            typename T_ForbiddenFaceView,
            typename T_ExitFaceView,
            typename T_CellPumpIntegralView,
            typename T_SamplePumpIntegralView>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            hase::core::DeviceMeshView const mesh,
            T_BetaVolumeView betaVolume,
            T_OriginXView originX,
            T_OriginYView originY,
            T_OriginZView originZ,
            T_DirectionXView directionX,
            T_DirectionYView directionY,
            T_DirectionZView directionZ,
            T_PowerView power,
            T_WavelengthView wavelength,
            T_SigmaAbsorptionView sigmaAbsorption,
            T_SigmaEmissionView sigmaEmission,
            T_CellView cell,
            T_ForbiddenFaceView forbiddenFace,
            T_ExitFaceView exitFace,
            T_CellPumpIntegralView cellPumpIntegral,
            T_SamplePumpIntegralView samplePumpIntegral,
            unsigned const rayCount) const
        {
            for(auto [ray] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{rayCount}))
            {
                hase::core::Point origin{originX[ray], originY[ray], originZ[ray]};
                hase::core::Point const direction{directionX[ray], directionY[ray], directionZ[ray]};
                unsigned tet = cell[ray];
                int forbidden = forbiddenFace[ray];
                double rayPower = power[ray];
                exitFace[ray] = -1;
                constexpr unsigned maxSteps = 10000u;
                for(unsigned step = 0u; step < maxSteps && rayPower != 0.0; ++step)
                {
                    auto const intersection
                        = hase::kernels::forward::nextFaceIntersection(mesh, tet, origin, direction, forbidden);
                    if(intersection.localFace < 0)
                    {
                        rayPower = 0.0;
                        break;
                    }
                    bool const gainCell = mesh.getCellType(tet) != mesh.claddingNumber;
                    double const gain = gainCell ? static_cast<double>(mesh.nTot)
                                                       * (betaVolume[tet] * (sigmaAbsorption[ray] + sigmaEmission[ray])
                                                          - sigmaAbsorption[ray])
                                                 : -mesh.claddingAbsorption;
                    double const exponent = gain * intersection.length;
                    if(!alpaka::math::isfinite(exponent) || exponent > 700.0)
                    {
                        rayPower = 0.0;
                        break;
                    }
                    double const nextPower = rayPower * alpaka::math::exp(exponent);
                    if(gainCell && mesh.nTot > 0.0f)
                    {
                        double const integral = (rayPower - nextPower) * wavelength[ray]
                                                / (planckConstant * speedOfLight * static_cast<double>(mesh.nTot));
                        alpaka::onAcc::atomicAdd(acc, &cellPumpIntegral[tet], integral);
                        if(mesh.samplePointsAreMeshPoints)
                        {
                            auto const midpoint = origin + direction * (0.5 * intersection.length);
                            auto const barycentric
                                = hase::kernels::forward::barycentricCoordinates(mesh, tet, midpoint);
                            for(unsigned vertex = 0u; vertex < mesh.numberOfCellVertices; ++vertex)
                                alpaka::onAcc::atomicAdd(
                                    acc,
                                    &samplePumpIntegral
                                        [mesh.cellPointIndices[tet * mesh.numberOfCellVertices + vertex]],
                                    integral * barycentric[vertex]);
                        }
                    }
                    rayPower = nextPower;
                    origin = hase::kernels::forward::advance(origin, direction, intersection.length);
                    auto const transition = hase::kernels::forward::transitionAcrossIntersection(
                        mesh,
                        tet,
                        intersection,
                        origin,
                        direction);
                    if(transition.status == hase::kernels::forward::Tet4TransitionStatus::failed)
                    {
                        rayPower = 0.0;
                        break;
                    }
                    if(transition.status == hase::kernels::forward::Tet4TransitionStatus::reachedBoundary)
                    {
                        exitFace[ray] = transition.boundaryFace;
                        break;
                    }
                    tet = transition.cell;
                    forbidden = transition.forbiddenFace;
                }
                originX[ray] = origin.x;
                originY[ray] = origin.y;
                originZ[ray] = origin.z;
                power[ray] = rayPower;
                cell[ray] = tet;
                forbiddenFace[ray] = forbidden;
            }
        }
    };

    struct ResetPumpRays
    {
        template<
            typename T_Acc,
            typename T_LaunchOriginX,
            typename T_LaunchOriginY,
            typename T_LaunchOriginZ,
            typename T_LaunchDirectionX,
            typename T_LaunchDirectionY,
            typename T_LaunchDirectionZ,
            typename T_LaunchPower,
            typename T_LaunchCell,
            typename T_LaunchForbiddenFace,
            typename T_OriginX,
            typename T_OriginY,
            typename T_OriginZ,
            typename T_DirectionX,
            typename T_DirectionY,
            typename T_DirectionZ,
            typename T_Power,
            typename T_Cell,
            typename T_ForbiddenFace,
            typename T_ExitFace>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            T_LaunchOriginX launchOriginX,
            T_LaunchOriginY launchOriginY,
            T_LaunchOriginZ launchOriginZ,
            T_LaunchDirectionX launchDirectionX,
            T_LaunchDirectionY launchDirectionY,
            T_LaunchDirectionZ launchDirectionZ,
            T_LaunchPower launchPower,
            T_LaunchCell launchCell,
            T_LaunchForbiddenFace launchForbiddenFace,
            T_OriginX originX,
            T_OriginY originY,
            T_OriginZ originZ,
            T_DirectionX directionX,
            T_DirectionY directionY,
            T_DirectionZ directionZ,
            T_Power power,
            T_Cell cell,
            T_ForbiddenFace forbiddenFace,
            T_ExitFace exitFace,
            unsigned const rayCount) const
        {
            for(auto [ray] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{rayCount}))
            {
                originX[ray] = launchOriginX[ray];
                originY[ray] = launchOriginY[ray];
                originZ[ray] = launchOriginZ[ray];
                directionX[ray] = launchDirectionX[ray];
                directionY[ray] = launchDirectionY[ray];
                directionZ[ray] = launchDirectionZ[ray];
                power[ray] = launchPower[ray];
                cell[ray] = launchCell[ray];
                forbiddenFace[ray] = launchForbiddenFace[ray];
                exitFace[ray] = -1;
            }
        }
    };

    template<
        typename T_Device,
        typename T_Executor,
        typename T_BetaBuffer,
        typename T_CellBuffer,
        typename T_SampleBuffer>
    PumpRayBatch tracePumpBatch(
        hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
        auto const& queue,
        hase::core::DeviceMeshView const mesh,
        T_BetaBuffer& betaVolume,
        T_CellBuffer& cellPumpIntegral,
        T_SampleBuffer& samplePumpIntegral,
        PumpRayBatch batch)
    {
        unsigned const count = static_cast<unsigned>(batch.size());
        if(count == 0u)
            return batch;
        auto originX = hase::alpakaUtils::toDevice(queue, batch.originX);
        auto originY = hase::alpakaUtils::toDevice(queue, batch.originY);
        auto originZ = hase::alpakaUtils::toDevice(queue, batch.originZ);
        auto directionX = hase::alpakaUtils::toDevice(queue, batch.directionX);
        auto directionY = hase::alpakaUtils::toDevice(queue, batch.directionY);
        auto directionZ = hase::alpakaUtils::toDevice(queue, batch.directionZ);
        auto power = hase::alpakaUtils::toDevice(queue, batch.power);
        auto wavelength = hase::alpakaUtils::toDevice(queue, batch.wavelength);
        auto sigmaA = hase::alpakaUtils::toDevice(queue, batch.sigmaAbsorption);
        auto sigmaE = hase::alpakaUtils::toDevice(queue, batch.sigmaEmission);
        auto cell = hase::alpakaUtils::toDevice(queue, batch.cell);
        auto forbiddenFace = hase::alpakaUtils::toDevice(queue, batch.forbiddenFace);
        auto exitFace = hase::alpakaUtils::toDevice(queue, batch.exitFace);
        auto frameSpec
            = hase::alpakaUtils::getFrameSpec<uint32_t>(devBundle.device, devBundle.executor, alpaka::Vec{count});
        queue.enqueue(
            frameSpec,
            alpaka::KernelBundle{
                TraceGeneralPump{},
                mesh,
                betaVolume,
                originX,
                originY,
                originZ,
                directionX,
                directionY,
                directionZ,
                power,
                wavelength,
                sigmaA,
                sigmaE,
                cell,
                forbiddenFace,
                exitFace,
                cellPumpIntegral,
                samplePumpIntegral,
                count});
        alpaka::onHost::wait(queue);
        auto copyBack = [&](auto const& deviceBuffer, auto& values)
        {
            auto host = alpaka::onHost::allocHostLike(deviceBuffer);
            alpaka::onHost::memcpy(queue, host, deviceBuffer);
            alpaka::onHost::wait(queue);
            std::copy_n(alpaka::onHost::data(host), values.size(), values.begin());
        };
        copyBack(originX, batch.originX);
        copyBack(originY, batch.originY);
        copyBack(originZ, batch.originZ);
        copyBack(directionX, batch.directionX);
        copyBack(directionY, batch.directionY);
        copyBack(directionZ, batch.directionZ);
        copyBack(power, batch.power);
        copyBack(cell, batch.cell);
        copyBack(forbiddenFace, batch.forbiddenFace);
        copyBack(exitFace, batch.exitFace);
        return batch;
    }

    struct RelayFrame
    {
        hase::core::Point origin, u, v, normal;
        std::vector<PumpBoundaryFace> faces;
    };

    [[nodiscard]] inline RelayFrame makeRelayFrame(hase::core::HostMesh const& mesh, std::vector<int> const& domains)
    {
        RelayFrame frame{};
        frame.faces = pumpBoundaryFaces(mesh, domains);
        if(frame.faces.empty())
            throw std::runtime_error("pump relay selected no exterior faces");
        double totalArea = 0.0;
        for(auto const& face : frame.faces)
        {
            frame.origin = frame.origin + face.centroid * face.area;
            frame.normal = frame.normal + face.normal * face.area;
            totalArea += face.area;
        }
        frame.origin = frame.origin * (1.0 / totalArea);
        frame.normal = hostNormalize(frame.normal);
        frame.u = hostNormalize(
            (frame.faces.front().vertices[1] - frame.faces.front().vertices[0])
            - frame.normal
                  * hase::core::dot(frame.faces.front().vertices[1] - frame.faces.front().vertices[0], frame.normal));
        if(frame.u.euclidLength() == 0.0)
            frame.u = perpendicular(frame.normal);
        frame.v = hase::core::cross(frame.normal, frame.u);
        double scale = 0.0;
        for(auto const& face : frame.faces)
            for(auto const& vertex : face.vertices)
                scale = std::max(scale, (vertex - frame.origin).euclidLength());
        for(auto const& face : frame.faces)
        {
            if(std::abs(hase::core::dot(face.centroid - frame.origin, frame.normal)) > 1.0e-8 * std::max(1.0, scale))
                throw std::runtime_error("pump relay surfaces must be coplanar");
        }
        return frame;
    }

    [[nodiscard]] inline bool pointInTriangle(
        hase::core::Point const point,
        PumpBoundaryFace const& face,
        hase::core::Point const u,
        hase::core::Point const v)
    {
        auto project = [&](hase::core::Point const p)
        { return std::array<double, 2u>{hase::core::dot(p, u), hase::core::dot(p, v)}; };
        auto const p = project(point);
        auto const a = project(face.vertices[0]);
        auto const b = project(face.vertices[1]);
        auto const c = project(face.vertices[2]);
        double const denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        if(std::abs(denominator) <= 1.0e-30)
            return false;
        double const wa = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denominator;
        double const wb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denominator;
        double const wc = 1.0 - wa - wb;
        return wa >= -1.0e-10 && wb >= -1.0e-10 && wc >= -1.0e-10;
    }

    [[nodiscard]] inline PumpRayBatch applyPumpRelay(
        hase::core::HostMesh const& mesh,
        PumpRayBatch const& exits,
        hase::core::PumpRelayParameters const& relay)
    {
        auto const exitFrame = makeRelayFrame(mesh, relay.exitSurfaces);
        auto const entryFrame = makeRelayFrame(mesh, relay.entrySurfaces);
        PumpRayBatch result;
        double const cosine = std::cos(relay.rotation);
        double const sine = std::sin(relay.rotation);
        for(std::size_t ray = 0u; ray < exits.size(); ++ray)
        {
            if(exits.exitFace[ray] < 0 || exits.power[ray] == 0.0)
                continue;
            unsigned const faceIndex
                = exits.cell[ray] * mesh.numberOfFacesPerCell + static_cast<unsigned>(exits.exitFace[ray]);
            int const domain = mesh.cellFaceBoundaries[faceIndex];
            if(!containsDomain(relay.exitSurfaces, domain))
                continue;
            hase::core::Point const position{exits.originX[ray], exits.originY[ray], exits.originZ[ray]};
            hase::core::Point const relative = position - exitFrame.origin;
            double u = hase::core::dot(relative, exitFrame.u) * (relay.flipU ? -1.0 : 1.0);
            double v = hase::core::dot(relative, exitFrame.v) * (relay.flipV ? -1.0 : 1.0);
            u *= relay.magnification;
            v *= relay.magnification;
            double const mappedU = cosine * u - sine * v + relay.offset[0];
            double const mappedV = sine * u + cosine * v + relay.offset[1];
            hase::core::Point const mappedPosition
                = entryFrame.origin + entryFrame.u * mappedU + entryFrame.v * mappedV;

            PumpBoundaryFace const* entryFace = nullptr;
            for(auto const& candidate : entryFrame.faces)
            {
                if(pointInTriangle(mappedPosition, candidate, entryFrame.u, entryFrame.v))
                {
                    entryFace = &candidate;
                    break;
                }
            }
            if(entryFace == nullptr)
                continue;

            hase::core::Point const oldDirection{exits.directionX[ray], exits.directionY[ray], exits.directionZ[ray]};
            double du = hase::core::dot(oldDirection, exitFrame.u) * (relay.flipU ? -1.0 : 1.0);
            double dv = hase::core::dot(oldDirection, exitFrame.v) * (relay.flipV ? -1.0 : 1.0);
            double const mappedDu = cosine * du - sine * dv + relay.tilt[0];
            double const mappedDv = sine * du + cosine * dv + relay.tilt[1];
            double const normalMagnitude = std::abs(hase::core::dot(oldDirection, exitFrame.normal));
            hase::core::Point const direction = hostNormalize(
                entryFrame.u * mappedDu + entryFrame.v * mappedDv - entryFrame.normal * normalMagnitude);

            result.originX.push_back(mappedPosition.x);
            result.originY.push_back(mappedPosition.y);
            result.originZ.push_back(mappedPosition.z);
            result.directionX.push_back(direction.x);
            result.directionY.push_back(direction.y);
            result.directionZ.push_back(direction.z);
            result.power.push_back(exits.power[ray] * relay.transmission);
            result.wavelength.push_back(exits.wavelength[ray]);
            result.sigmaAbsorption.push_back(exits.sigmaAbsorption[ray]);
            result.sigmaEmission.push_back(exits.sigmaEmission[ray]);
            result.cell.push_back(entryFace->cell);
            result.forbiddenFace.push_back(static_cast<int>(entryFace->localFace));
            result.exitFace.push_back(-1);
        }
        return result;
    }

    struct ApplyPumpRelayOnDevice
    {
        hase::core::Point exitOrigin;
        hase::core::Point exitU;
        hase::core::Point exitV;
        hase::core::Point exitNormal;
        hase::core::Point entryOrigin;
        hase::core::Point entryU;
        hase::core::Point entryV;
        hase::core::Point entryNormal;
        bool flipU = false;
        bool flipV = false;
        double cosine = 1.0;
        double sine = 0.0;
        double offsetU = 0.0;
        double offsetV = 0.0;
        double tiltU = 0.0;
        double tiltV = 0.0;
        double magnification = 1.0;
        double transmission = 1.0;

        template<typename T_ExitDomains>
        ALPAKA_FN_ACC bool acceptsExitDomain(
            T_ExitDomains const& exitDomains,
            unsigned const exitDomainCount,
            int const domain) const
        {
            for(unsigned index = 0u; index < exitDomainCount; ++index)
                if(exitDomains[index] == domain)
                    return true;
            return false;
        }

        ALPAKA_FN_ACC bool containsEntryPoint(hase::core::Point const point, PumpBoundaryFace const& face) const
        {
            double const pointU = hase::core::dot(point, entryU);
            double const pointV = hase::core::dot(point, entryV);
            double const aU = hase::core::dot(face.vertices[0], entryU);
            double const aV = hase::core::dot(face.vertices[0], entryV);
            double const bU = hase::core::dot(face.vertices[1], entryU);
            double const bV = hase::core::dot(face.vertices[1], entryV);
            double const cU = hase::core::dot(face.vertices[2], entryU);
            double const cV = hase::core::dot(face.vertices[2], entryV);
            double const denominator = (bV - cV) * (aU - cU) + (cU - bU) * (aV - cV);
            if(alpaka::math::abs(denominator) <= 1.0e-30)
                return false;
            double const weightA = ((bV - cV) * (pointU - cU) + (cU - bU) * (pointV - cV)) / denominator;
            double const weightB = ((cV - aV) * (pointU - cU) + (aU - cU) * (pointV - cV)) / denominator;
            double const weightC = 1.0 - weightA - weightB;
            return weightA >= -1.0e-10 && weightB >= -1.0e-10 && weightC >= -1.0e-10;
        }

        ALPAKA_FN_ACC hase::core::Point normalize(hase::core::Point const value) const
        {
            double const length = value.euclidLength();
            return length > 0.0 ? value * (1.0 / length) : hase::core::Point{0.0, 0.0, 0.0};
        }

        template<
            typename T_Acc,
            typename T_EntryFaces,
            typename T_ExitDomains,
            typename T_OriginX,
            typename T_OriginY,
            typename T_OriginZ,
            typename T_DirectionX,
            typename T_DirectionY,
            typename T_DirectionZ,
            typename T_Power,
            typename T_Cell,
            typename T_ForbiddenFace,
            typename T_ExitFace>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            hase::core::DeviceMeshView const mesh,
            T_EntryFaces const& entryFaces,
            unsigned const entryFaceCount,
            T_ExitDomains const& exitDomains,
            unsigned const exitDomainCount,
            T_OriginX originX,
            T_OriginY originY,
            T_OriginZ originZ,
            T_DirectionX directionX,
            T_DirectionY directionY,
            T_DirectionZ directionZ,
            T_Power power,
            T_Cell cell,
            T_ForbiddenFace forbiddenFace,
            T_ExitFace exitFace,
            unsigned const rayCount) const
        {
            for(auto [ray] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{rayCount}))
            {
                if(exitFace[ray] < 0 || power[ray] == 0.0)
                {
                    power[ray] = 0.0;
                    exitFace[ray] = -1;
                    continue;
                }
                unsigned const faceIndex
                    = cell[ray] * mesh.numberOfFacesPerCell + static_cast<unsigned>(exitFace[ray]);
                int const domain = mesh.cellFaceBoundaries[faceIndex];
                if(!acceptsExitDomain(exitDomains, exitDomainCount, domain))
                {
                    power[ray] = 0.0;
                    exitFace[ray] = -1;
                    continue;
                }

                hase::core::Point const position{originX[ray], originY[ray], originZ[ray]};
                hase::core::Point const relative = position - exitOrigin;
                double sourceU = hase::core::dot(relative, exitU) * (flipU ? -1.0 : 1.0);
                double sourceV = hase::core::dot(relative, exitV) * (flipV ? -1.0 : 1.0);
                sourceU *= magnification;
                sourceV *= magnification;
                double const mappedU = cosine * sourceU - sine * sourceV + offsetU;
                double const mappedV = sine * sourceU + cosine * sourceV + offsetV;
                hase::core::Point const mappedPosition = entryOrigin + entryU * mappedU + entryV * mappedV;

                int selectedEntryFace = -1;
                for(unsigned candidate = 0u; candidate < entryFaceCount; ++candidate)
                {
                    if(containsEntryPoint(mappedPosition, entryFaces[candidate]))
                    {
                        selectedEntryFace = static_cast<int>(candidate);
                        break;
                    }
                }
                if(selectedEntryFace < 0)
                {
                    power[ray] = 0.0;
                    exitFace[ray] = -1;
                    continue;
                }

                hase::core::Point const oldDirection{directionX[ray], directionY[ray], directionZ[ray]};
                double const sourceDirectionU = hase::core::dot(oldDirection, exitU) * (flipU ? -1.0 : 1.0);
                double const sourceDirectionV = hase::core::dot(oldDirection, exitV) * (flipV ? -1.0 : 1.0);
                double const mappedDirectionU = cosine * sourceDirectionU - sine * sourceDirectionV + tiltU;
                double const mappedDirectionV = sine * sourceDirectionU + cosine * sourceDirectionV + tiltV;
                double const normalMagnitude = alpaka::math::abs(hase::core::dot(oldDirection, exitNormal));
                hase::core::Point const direction
                    = normalize(entryU * mappedDirectionU + entryV * mappedDirectionV - entryNormal * normalMagnitude);
                auto const& entryFace = entryFaces[static_cast<unsigned>(selectedEntryFace)];
                originX[ray] = mappedPosition.x;
                originY[ray] = mappedPosition.y;
                originZ[ray] = mappedPosition.z;
                directionX[ray] = direction.x;
                directionY[ray] = direction.y;
                directionZ[ray] = direction.z;
                power[ray] *= transmission;
                cell[ray] = entryFace.cell;
                forbiddenFace[ray] = static_cast<int>(entryFace.localFace);
                exitFace[ray] = -1;
            }
        }
    };

    struct NormalizePumpRate
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            hase::core::DeviceMeshView const mesh,
            auto cellIntegral,
            auto lumpedVolume,
            auto sampleRate) const
        {
            for(auto [sample] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{mesh.numberOfSamples}))
            {
                if(mesh.samplePointsAreMeshPoints)
                    sampleRate[sample] = lumpedVolume[sample] > 0.0 ? sampleRate[sample] / lumpedVolume[sample] : 0.0;
                else
                    sampleRate[sample] = cellIntegral[sample] / mesh.getCellVolume(sample);
            }
        }
    };

    class PumpTimingCsv
    {
    public:
        PumpTimingCsv()
        {
            if(auto const* path = std::getenv("HASE_PUMP_TIMING_CSV"))
            {
                m_stream.open(path);
                if(!m_stream)
                    throw std::runtime_error("failed to open HASE_PUMP_TIMING_CSV");
                m_stream << "invocation,source,relay,phase,elapsed_seconds\n";
            }
        }

        [[nodiscard]] bool enabled() const
        {
            return m_stream.is_open();
        }

        [[nodiscard]] std::uint64_t nextInvocation()
        {
            return ++m_invocation;
        }

        void record(
            std::uint64_t const invocation,
            int const source,
            int const relay,
            std::string const& phase,
            double const elapsedSeconds)
        {
            if(!enabled())
                return;
            m_stream << invocation << ',' << source << ',' << relay << ',' << phase << ',' << std::setprecision(17)
                     << elapsedSeconds << '\n';
            m_stream.flush();
        }

    private:
        std::ofstream m_stream;
        std::uint64_t m_invocation = 0u;
    };

    template<typename T_Device>
    class GeneralPumpWorkspace
    {
        using T_DoubleBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1}));
        using T_UnsignedBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<unsigned>(std::declval<T_Device&>(), std::size_t{1}));
        using T_IntBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<int>(std::declval<T_Device&>(), std::size_t{1}));
        using T_FaceBuffer
            = ALPAKA_TYPEOF(alpaka::onHost::alloc<PumpBoundaryFace>(std::declval<T_Device&>(), std::size_t{1}));

        struct RelayWorkspace
        {
            RelayWorkspace(
                auto const& queue,
                hase::core::HostMesh const& mesh,
                hase::core::PumpRelayParameters const& relay)
                : parameters(relay)
                , exitFrame(makeRelayFrame(mesh, relay.exitSurfaces))
                , entryFrame(makeRelayFrame(mesh, relay.entrySurfaces))
                , entryFaces(hase::alpakaUtils::toDevice(queue, entryFrame.faces))
                , exitDomains(hase::alpakaUtils::toDevice(queue, parameters.exitSurfaces))
            {
            }

            [[nodiscard]] ApplyPumpRelayOnDevice kernel() const
            {
                return ApplyPumpRelayOnDevice{
                    exitFrame.origin,
                    exitFrame.u,
                    exitFrame.v,
                    exitFrame.normal,
                    entryFrame.origin,
                    entryFrame.u,
                    entryFrame.v,
                    entryFrame.normal,
                    parameters.flipU,
                    parameters.flipV,
                    std::cos(parameters.rotation),
                    std::sin(parameters.rotation),
                    parameters.offset[0],
                    parameters.offset[1],
                    parameters.tilt[0],
                    parameters.tilt[1],
                    parameters.magnification,
                    parameters.transmission};
            }

            hase::core::PumpRelayParameters parameters;
            RelayFrame exitFrame;
            RelayFrame entryFrame;
            T_FaceBuffer entryFaces;
            T_IntBuffer exitDomains;
        };

        struct SourceWorkspace
        {
            SourceWorkspace(
                T_Device& device,
                auto const& queue,
                hase::core::HostMesh const& mesh,
                hase::core::PumpSourceParameters const& source,
                PumpRayBatch const& launch)
                : rayCount(static_cast<unsigned>(launch.size()))
                , launchOriginX(hase::alpakaUtils::toDevice(queue, launch.originX))
                , launchOriginY(hase::alpakaUtils::toDevice(queue, launch.originY))
                , launchOriginZ(hase::alpakaUtils::toDevice(queue, launch.originZ))
                , launchDirectionX(hase::alpakaUtils::toDevice(queue, launch.directionX))
                , launchDirectionY(hase::alpakaUtils::toDevice(queue, launch.directionY))
                , launchDirectionZ(hase::alpakaUtils::toDevice(queue, launch.directionZ))
                , launchPower(hase::alpakaUtils::toDevice(queue, launch.power))
                , wavelength(hase::alpakaUtils::toDevice(queue, launch.wavelength))
                , sigmaAbsorption(hase::alpakaUtils::toDevice(queue, launch.sigmaAbsorption))
                , sigmaEmission(hase::alpakaUtils::toDevice(queue, launch.sigmaEmission))
                , launchCell(hase::alpakaUtils::toDevice(queue, launch.cell))
                , launchForbiddenFace(hase::alpakaUtils::toDevice(queue, launch.forbiddenFace))
                , originX(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(rayCount)))
                , originY(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(rayCount)))
                , originZ(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(rayCount)))
                , directionX(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(rayCount)))
                , directionY(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(rayCount)))
                , directionZ(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(rayCount)))
                , power(alpaka::onHost::alloc<double>(device, static_cast<std::size_t>(rayCount)))
                , cell(alpaka::onHost::alloc<unsigned>(device, static_cast<std::size_t>(rayCount)))
                , forbiddenFace(alpaka::onHost::alloc<int>(device, static_cast<std::size_t>(rayCount)))
                , exitFace(alpaka::onHost::alloc<int>(device, static_cast<std::size_t>(rayCount)))
            {
                relays.reserve(source.relays.size());
                for(auto const& relay : source.relays)
                    relays.emplace_back(std::make_unique<RelayWorkspace>(queue, mesh, relay));
            }

            unsigned rayCount;
            T_DoubleBuffer launchOriginX, launchOriginY, launchOriginZ;
            T_DoubleBuffer launchDirectionX, launchDirectionY, launchDirectionZ;
            T_DoubleBuffer launchPower, wavelength, sigmaAbsorption, sigmaEmission;
            T_UnsignedBuffer launchCell;
            T_IntBuffer launchForbiddenFace;
            T_DoubleBuffer originX, originY, originZ;
            T_DoubleBuffer directionX, directionY, directionZ;
            T_DoubleBuffer power;
            T_UnsignedBuffer cell;
            T_IntBuffer forbiddenFace, exitFace;
            std::vector<std::unique_ptr<RelayWorkspace>> relays;
        };

    public:
        GeneralPumpWorkspace(
            T_Device& device,
            auto const& queue,
            hase::core::HostMesh const& mesh,
            hase::core::PumpParameters const& pump)
        {
            m_sources.reserve(pump.sources.size());
            for(std::size_t sourceIndex = 0u; sourceIndex < pump.sources.size(); ++sourceIndex)
            {
                auto const sampleStarted = std::chrono::steady_clock::now();
                PumpRayBatch const launch = samplePumpSource(
                    mesh,
                    pump.sources[sourceIndex],
                    pump.rayCount,
                    pump.rngSeed + static_cast<std::uint32_t>(sourceIndex));
                m_timings.record(
                    0u,
                    static_cast<int>(sourceIndex),
                    -1,
                    "prepare_sample",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - sampleStarted).count());
                auto const uploadStarted = std::chrono::steady_clock::now();
                m_sources.emplace_back(
                    std::make_unique<SourceWorkspace>(device, queue, mesh, pump.sources[sourceIndex], launch));
                m_timings.record(
                    0u,
                    static_cast<int>(sourceIndex),
                    -1,
                    "prepare_upload",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - uploadStarted).count());
            }
        }

        void enqueue(
            auto& devBundle,
            auto const& queue,
            hase::core::DeviceMeshView const mesh,
            auto& betaVolume,
            auto& cellPumpIntegral,
            auto& lumpedVolume,
            auto& sampleRate)
        {
            std::uint64_t const invocation = m_timings.nextInvocation();
            measure(
                queue,
                invocation,
                -1,
                -1,
                "clear",
                [&]
                {
                    alpaka::onHost::fill(
                        queue,
                        cellPumpIntegral,
                        0.0,
                        alpaka::Vec{static_cast<std::size_t>(mesh.numberOfCells)});
                    alpaka::onHost::fill(
                        queue,
                        sampleRate,
                        0.0,
                        alpaka::Vec{static_cast<std::size_t>(mesh.numberOfSamples)});
                });

            for(std::size_t sourceIndex = 0u; sourceIndex < m_sources.size(); ++sourceIndex)
            {
                auto& source = *m_sources[sourceIndex];
                int const sourceNumber = static_cast<int>(sourceIndex);
                measure(
                    queue,
                    invocation,
                    sourceNumber,
                    -1,
                    "reset",
                    [&]
                    {
                        auto frameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
                            devBundle.device,
                            devBundle.executor,
                            alpaka::Vec{source.rayCount});
                        queue.enqueue(
                            frameSpec,
                            alpaka::KernelBundle{
                                ResetPumpRays{},
                                source.launchOriginX,
                                source.launchOriginY,
                                source.launchOriginZ,
                                source.launchDirectionX,
                                source.launchDirectionY,
                                source.launchDirectionZ,
                                source.launchPower,
                                source.launchCell,
                                source.launchForbiddenFace,
                                source.originX,
                                source.originY,
                                source.originZ,
                                source.directionX,
                                source.directionY,
                                source.directionZ,
                                source.power,
                                source.cell,
                                source.forbiddenFace,
                                source.exitFace,
                                source.rayCount});
                    });
                measure(
                    queue,
                    invocation,
                    sourceNumber,
                    -1,
                    "direct_trace",
                    [&] { enqueueTrace(devBundle, queue, mesh, betaVolume, cellPumpIntegral, sampleRate, source); });

                for(std::size_t relayIndex = 0u; relayIndex < source.relays.size(); ++relayIndex)
                {
                    auto& relay = *source.relays[relayIndex];
                    int const relayNumber = static_cast<int>(relayIndex);
                    measure(
                        queue,
                        invocation,
                        sourceNumber,
                        relayNumber,
                        "relay_map",
                        [&]
                        {
                            auto frameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
                                devBundle.device,
                                devBundle.executor,
                                alpaka::Vec{source.rayCount});
                            queue.enqueue(
                                frameSpec,
                                alpaka::KernelBundle{
                                    relay.kernel(),
                                    mesh,
                                    relay.entryFaces,
                                    static_cast<unsigned>(relay.entryFrame.faces.size()),
                                    relay.exitDomains,
                                    static_cast<unsigned>(relay.parameters.exitSurfaces.size()),
                                    source.originX,
                                    source.originY,
                                    source.originZ,
                                    source.directionX,
                                    source.directionY,
                                    source.directionZ,
                                    source.power,
                                    source.cell,
                                    source.forbiddenFace,
                                    source.exitFace,
                                    source.rayCount});
                        });
                    measure(
                        queue,
                        invocation,
                        sourceNumber,
                        relayNumber,
                        "relay_trace",
                        [&]
                        { enqueueTrace(devBundle, queue, mesh, betaVolume, cellPumpIntegral, sampleRate, source); });
                }
            }

            measure(
                queue,
                invocation,
                -1,
                -1,
                "normalize",
                [&]
                {
                    auto frameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
                        devBundle.device,
                        devBundle.executor,
                        alpaka::Vec{mesh.numberOfSamples});
                    queue.enqueue(
                        frameSpec,
                        alpaka::KernelBundle{NormalizePumpRate{}, mesh, cellPumpIntegral, lumpedVolume, sampleRate});
                });
            alpaka::onHost::wait(queue);
        }

    private:
        void enqueueTrace(
            auto& devBundle,
            auto const& queue,
            hase::core::DeviceMeshView const mesh,
            auto& betaVolume,
            auto& cellPumpIntegral,
            auto& sampleRate,
            SourceWorkspace& source)
        {
            auto frameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
                devBundle.device,
                devBundle.executor,
                alpaka::Vec{source.rayCount});
            queue.enqueue(
                frameSpec,
                alpaka::KernelBundle{
                    TraceGeneralPump{},
                    mesh,
                    betaVolume,
                    source.originX,
                    source.originY,
                    source.originZ,
                    source.directionX,
                    source.directionY,
                    source.directionZ,
                    source.power,
                    source.wavelength,
                    source.sigmaAbsorption,
                    source.sigmaEmission,
                    source.cell,
                    source.forbiddenFace,
                    source.exitFace,
                    cellPumpIntegral,
                    sampleRate,
                    source.rayCount});
        }

        template<typename T_Function>
        void measure(
            auto const& queue,
            std::uint64_t const invocation,
            int const source,
            int const relay,
            std::string const& phase,
            T_Function&& function)
        {
            if(!m_timings.enabled())
            {
                std::forward<T_Function>(function)();
                return;
            }
            alpaka::onHost::wait(queue);
            auto const started = std::chrono::steady_clock::now();
            std::forward<T_Function>(function)();
            alpaka::onHost::wait(queue);
            m_timings.record(
                invocation,
                source,
                relay,
                phase,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        }

        PumpTimingCsv m_timings;
        std::vector<std::unique_ptr<SourceWorkspace>> m_sources;
    };

    template<
        typename T_Device,
        typename T_Executor,
        typename T_BetaBuffer,
        typename T_CellBuffer,
        typename T_LumpedBuffer,
        typename T_SampleBuffer>
    void enqueueGeneralPump(
        hase::alpakaUtils::DevBundle<T_Device, T_Executor>& devBundle,
        auto const& queue,
        hase::core::HostMesh const& hostMesh,
        hase::core::DeviceMeshView const mesh,
        hase::core::PumpParameters const& pump,
        T_BetaBuffer& betaVolume,
        T_CellBuffer& cellPumpIntegral,
        T_LumpedBuffer& lumpedVolume,
        T_SampleBuffer& sampleRate)
    {
        alpaka::onHost::fill(queue, cellPumpIntegral, 0.0, alpaka::Vec{static_cast<std::size_t>(mesh.numberOfCells)});
        alpaka::onHost::fill(queue, sampleRate, 0.0, alpaka::Vec{static_cast<std::size_t>(mesh.numberOfSamples)});
        alpaka::onHost::wait(queue);
        for(std::size_t sourceIndex = 0u; sourceIndex < pump.sources.size(); ++sourceIndex)
        {
            auto const& source = pump.sources[sourceIndex];
            PumpRayBatch rays = samplePumpSource(
                hostMesh,
                source,
                pump.rayCount,
                pump.rngSeed + static_cast<std::uint32_t>(sourceIndex));
            rays = tracePumpBatch(devBundle, queue, mesh, betaVolume, cellPumpIntegral, sampleRate, std::move(rays));
            for(auto const& relay : source.relays)
            {
                rays = applyPumpRelay(hostMesh, rays, relay);
                rays = tracePumpBatch(
                    devBundle,
                    queue,
                    mesh,
                    betaVolume,
                    cellPumpIntegral,
                    sampleRate,
                    std::move(rays));
            }
        }
        auto sampleFrameSpec = hase::alpakaUtils::getFrameSpec<uint32_t>(
            devBundle.device,
            devBundle.executor,
            alpaka::Vec{mesh.numberOfSamples});
        queue.enqueue(
            sampleFrameSpec,
            alpaka::KernelBundle{NormalizePumpRate{}, mesh, cellPumpIntegral, lumpedVolume, sampleRate});
        alpaka::onHost::wait(queue);
    }
} // namespace hase::kernels
