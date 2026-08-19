#include <backend/legacy/LegacyBackendConverter.hpp>
#include <core/mesh.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace hase::backend::legacy
{
    namespace
    {
        constexpr std::size_t coordinates = 3u;
        constexpr std::size_t verticesPerCell = 4u;
        constexpr std::size_t facesPerCell = 4u;
        constexpr std::size_t verticesPerFace = 3u;

        [[noreturn]] void invalid(std::string const& message)
        {
            throw std::runtime_error("legacy backend conversion: " + message);
        }

        template<typename T>
        T narrow(std::uint64_t value, std::string const& name)
        {
            if(value > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
                invalid(name + " is outside the legacy backend range");
            return static_cast<T>(value);
        }

        template<typename T>
        void requireSize(std::vector<T> const& values, std::size_t expected, std::string const& name)
        {
            if(values.size() != expected)
                invalid(
                    name + " has " + std::to_string(values.size()) + " values; expected " + std::to_string(expected));
        }

        struct Shard
        {
            std::shared_ptr<VolumeTopology> topology;
            std::vector<std::uint8_t> mask;
        };

        std::vector<Shard> shards(Domain const& domain)
        {
            if(domain.masks.offsets.size() != domain.topologies.size() + 1u || domain.masks.offsets.front() != 0u
               || domain.masks.offsets.back() != domain.masks.values.size())
                invalid("domain mask offsets do not match its topology references");
            std::vector<Shard> result;
            result.reserve(domain.topologies.size());
            for(std::size_t index = 0u; index < domain.topologies.size(); ++index)
            {
                auto const begin = domain.masks.offsets[index];
                auto const end = domain.masks.offsets[index + 1u];
                if(begin > end || end > domain.masks.values.size())
                    invalid("domain mask offsets are not monotonic");
                result.push_back(
                    {domain.topologies[index],
                     {domain.masks.values.begin() + static_cast<std::ptrdiff_t>(begin),
                      domain.masks.values.begin() + static_cast<std::ptrdiff_t>(end)}});
            }
            return result;
        }

        std::size_t cellCount(VolumeTopology const& topology)
        {
            if(topology.cellPointIndices.values.size() % verticesPerCell != 0u)
                invalid("volume topology connectivity is not Tet4");
            auto const count = topology.cellPointIndices.values.size() / verticesPerCell;
            requireSize(topology.cellTypes.values, count, "volume topology cellTypes");
            requireSize(topology.cellDomains.values, count, "volume topology cellDomains");
            requireSize(
                topology.facePointIndices.values,
                count * facesPerCell * verticesPerFace,
                "volume topology facePointIndices");
            requireSize(topology.neighborCells.values, count * facesPerCell, "volume topology neighborCells");
            requireSize(
                topology.neighborLocalFaces.values,
                count * facesPerCell,
                "volume topology neighborLocalFaces");
            requireSize(topology.faceBoundaries.values, count * facesPerCell, "volume topology faceBoundaries");
            requireSize(topology.cellCenters.values, count * coordinates, "volume topology cellCenters");
            requireSize(topology.cellVolumes.values, count, "volume topology cellVolumes");
            return count;
        }

        std::size_t pointCount(VolumeTopology const& topology)
        {
            if(topology.points.values.size() % coordinates != 0u)
                invalid("volume topology points are not three-dimensional");
            return topology.points.values.size() / coordinates;
        }

        template<typename T>
        T const& soa2(std::vector<T> const& values, std::size_t items, std::size_t component, std::size_t item)
        {
            return values.at(component * items + item);
        }

        template<typename T>
        T const& soa3(
            std::vector<T> const& values,
            std::size_t middle,
            std::size_t items,
            std::size_t component,
            std::size_t middleIndex,
            std::size_t item)
        {
            return values.at((component * middle + middleIndex) * items + item);
        }

        struct TopologyMap
        {
            std::shared_ptr<VolumeTopology> topology;
            std::vector<std::uint8_t> selected;
            std::vector<int> cells;
            std::vector<int> points;
        };

        struct Assembly
        {
            std::vector<TopologyMap> topologies;
            std::unordered_map<VolumeTopology const*, std::size_t> lookup;
            std::vector<unsigned> cellPointIndices;
            std::vector<unsigned> cellTypes;
            std::vector<int> cellFaces;
            std::vector<int> neighborCells;
            std::vector<int> neighborLocalFaces;
            std::vector<int> faceBoundaries;
            std::vector<float> cellVolumes;
            std::vector<double> points;
            std::vector<double> samplePoints;
            std::vector<double> cellCenters;
        };

        TopologyMap& topologyMap(Assembly& assembly, std::shared_ptr<VolumeTopology> const& topology)
        {
            if(!topology)
                invalid("domain references a null topology");
            if(auto const found = assembly.lookup.find(topology.get()); found != assembly.lookup.end())
                return assembly.topologies[found->second];
            auto const count = cellCount(*topology);
            auto const index = assembly.topologies.size();
            assembly.lookup.emplace(topology.get(), index);
            assembly.topologies.push_back(
                {topology, std::vector<std::uint8_t>(count, 0u), std::vector<int>(count, -1), {}});
            return assembly.topologies.back();
        }

        void selectDomain(Assembly& assembly, Domain const& domain, bool rejectOverlap)
        {
            if(domain.entityKind != "volume")
                invalid("optical components must reference volume domains");
            for(auto& shard : shards(domain))
            {
                auto& mapping = topologyMap(assembly, shard.topology);
                requireSize(shard.mask, mapping.selected.size(), "volume domain mask");
                for(std::size_t cell = 0u; cell < shard.mask.size(); ++cell)
                {
                    if(!shard.mask[cell])
                        continue;
                    if(rejectOverlap && mapping.selected[cell])
                        invalid("optical component volume domains overlap");
                    mapping.selected[cell] = 1u;
                }
            }
        }

        std::vector<double> structureOfArrays(std::vector<double> const& arrayOfStructures)
        {
            if(arrayOfStructures.size() % coordinates != 0u)
                invalid("coordinate array is not three-dimensional");
            auto const count = arrayOfStructures.size() / coordinates;
            std::vector<double> result(arrayOfStructures.size());
            for(std::size_t item = 0u; item < count; ++item)
                for(std::size_t coordinate = 0u; coordinate < coordinates; ++coordinate)
                    result[coordinate * count + item] = arrayOfStructures[item * coordinates + coordinate];
            return result;
        }

        void buildAssemblyArrays(Assembly& assembly)
        {
            std::vector<double> pointsAoS;
            std::vector<double> centersAoS;
            std::size_t nextCell = 0u;
            std::size_t nextPoint = 0u;

            for(auto& mapping : assembly.topologies)
            {
                auto const& topology = *mapping.topology;
                auto const cells = cellCount(topology);
                auto const points = pointCount(topology);
                mapping.points.assign(points, -1);

                std::vector<std::uint8_t> used(points, 0u);
                bool const complete = std::all_of(
                    mapping.selected.begin(),
                    mapping.selected.end(),
                    [](auto value) { return value != 0u; });
                if(complete)
                    std::fill(used.begin(), used.end(), 1u);
                else
                {
                    for(std::size_t cell = 0u; cell < cells; ++cell)
                        if(mapping.selected[cell])
                            for(std::size_t vertex = 0u; vertex < verticesPerCell; ++vertex)
                            {
                                auto const point = soa2(topology.cellPointIndices.values, cells, vertex, cell);
                                if(point >= points)
                                    invalid("volume topology connectivity references an invalid point");
                                used[point] = 1u;
                            }
                }
                for(std::size_t point = 0u; point < points; ++point)
                {
                    if(!used[point])
                        continue;
                    mapping.points[point] = static_cast<int>(nextPoint++);
                    pointsAoS.insert(
                        pointsAoS.end(),
                        {soa2(topology.points.values, points, 0u, point),
                         soa2(topology.points.values, points, 1u, point),
                         soa2(topology.points.values, points, 2u, point)});
                }
                for(std::size_t cell = 0u; cell < cells; ++cell)
                    if(mapping.selected[cell])
                        mapping.cells[cell] = static_cast<int>(nextCell++);
            }

            for(auto const& mapping : assembly.topologies)
            {
                auto const& topology = *mapping.topology;
                auto const cells = mapping.selected.size();
                for(std::size_t cell = 0u; cell < cells; ++cell)
                {
                    if(!mapping.selected[cell])
                        continue;
                    for(std::size_t vertex = 0u; vertex < verticesPerCell; ++vertex)
                    {
                        auto const source = soa2(topology.cellPointIndices.values, cells, vertex, cell);
                        auto const target = mapping.points.at(source);
                        if(target < 0)
                            invalid("selected cell references an unselected point");
                        assembly.cellPointIndices.push_back(static_cast<unsigned>(target));
                    }
                    assembly.cellTypes.push_back(topology.cellTypes.values[cell]);
                    for(std::size_t face = 0u; face < facesPerCell; ++face)
                    {
                        for(std::size_t vertex = 0u; vertex < verticesPerFace; ++vertex)
                        {
                            auto const source
                                = soa3(topology.facePointIndices.values, facesPerCell, cells, vertex, face, cell);
                            if(source < 0 || static_cast<std::size_t>(source) >= mapping.points.size()
                               || mapping.points[source] < 0)
                                invalid("selected cell face references an invalid point");
                            assembly.cellFaces.push_back(mapping.points[source]);
                        }
                        auto const sourceNeighbor = soa2(topology.neighborCells.values, cells, face, cell);
                        int targetNeighbor = -1;
                        if(sourceNeighbor >= 0 && static_cast<std::size_t>(sourceNeighbor) < mapping.cells.size())
                            targetNeighbor = mapping.cells[sourceNeighbor];
                        assembly.neighborCells.push_back(targetNeighbor);
                        assembly.neighborLocalFaces.push_back(
                            targetNeighbor < 0 ? -1 : soa2(topology.neighborLocalFaces.values, cells, face, cell));
                        auto boundary = soa2(topology.faceBoundaries.values, cells, face, cell);
                        if(targetNeighbor < 0 && boundary == 0)
                            boundary = -1;
                        assembly.faceBoundaries.push_back(boundary);
                    }
                    assembly.cellVolumes.push_back(static_cast<float>(topology.cellVolumes.values[cell]));
                    centersAoS.insert(
                        centersAoS.end(),
                        {soa2(topology.cellCenters.values, cells, 0u, cell),
                         soa2(topology.cellCenters.values, cells, 1u, cell),
                         soa2(topology.cellCenters.values, cells, 2u, cell)});
                }
            }
            if(assembly.cellTypes.empty())
                invalid("optical assembly is empty");
            assembly.points = structureOfArrays(pointsAoS);
            assembly.cellCenters = structureOfArrays(centersAoS);
            assembly.samplePoints = assembly.cellCenters;
        }

        std::vector<unsigned> domainCells(Assembly const& assembly, Domain const& domain)
        {
            if(domain.entityKind != "volume")
                invalid("cell assignment references a non-volume domain");
            std::vector<unsigned> result;
            for(auto const& shard : shards(domain))
            {
                auto const found = assembly.lookup.find(shard.topology.get());
                if(found == assembly.lookup.end())
                    invalid("domain contains cells outside the optical assembly");
                auto const& mapping = assembly.topologies[found->second];
                requireSize(shard.mask, mapping.cells.size(), "volume domain mask");
                for(std::size_t cell = 0u; cell < shard.mask.size(); ++cell)
                {
                    if(!shard.mask[cell])
                        continue;
                    if(mapping.cells[cell] < 0)
                        invalid("domain contains cells outside the optical assembly");
                    result.push_back(static_cast<unsigned>(mapping.cells[cell]));
                }
            }
            return result;
        }

        bool sameDomain(Domain const& left, Domain const& right)
        {
            if(left.entityKind != right.entityKind || left.topologies.size() != right.topologies.size()
               || left.masks.offsets != right.masks.offsets || left.masks.values != right.masks.values)
                return false;
            for(std::size_t index = 0u; index < left.topologies.size(); ++index)
                if(left.topologies[index].get() != right.topologies[index].get())
                    return false;
            return true;
        }

        class SurfaceDomains
        {
        public:
            explicit SurfaceDomains(Assembly& assembly) : m_assembly(assembly)
            {
                for(auto boundary : assembly.faceBoundaries)
                    if(boundary > 0)
                        m_next = std::max(m_next, boundary + 1);
            }

            int assign(std::shared_ptr<Domain> const& domain)
            {
                if(!domain || domain->entityKind != "surface")
                    invalid("surface assignment references a non-surface domain");
                for(auto const& [known, identifier] : m_domains)
                    if(sameDomain(*known, *domain))
                        return identifier;
                int const identifier = m_next++;
                bool any = false;
                for(auto const& shard : shards(*domain))
                {
                    auto const found = m_assembly.lookup.find(shard.topology.get());
                    if(found == m_assembly.lookup.end())
                        invalid("surface domain is outside the optical assembly");
                    auto const& mapping = m_assembly.topologies[found->second];
                    requireSize(shard.mask, mapping.cells.size() * facesPerCell, "surface domain mask");
                    for(std::size_t index = 0u; index < shard.mask.size(); ++index)
                    {
                        if(!shard.mask[index])
                            continue;
                        auto const cell = index / facesPerCell;
                        auto const face = index % facesPerCell;
                        if(mapping.cells[cell] < 0)
                            invalid("surface domain is outside the selected optical assembly");
                        m_assembly.faceBoundaries[static_cast<std::size_t>(mapping.cells[cell]) * facesPerCell + face]
                            = identifier;
                        any = true;
                    }
                }
                if(!any)
                    invalid("surface domain is empty");
                m_domains.emplace_back(domain, identifier);
                return identifier;
            }

        private:
            Assembly& m_assembly;
            int m_next = 1;
            std::vector<std::pair<std::shared_ptr<Domain>, int>> m_domains;
        };

        std::vector<double> excitation(Simulation const& simulation, Assembly const& assembly)
        {
            if(!simulation.excitationState || !simulation.gainMedium)
                invalid("simulation has no excitation state or gain medium");
            std::vector<std::uint8_t> active(assembly.cellTypes.size(), 0u);
            for(auto const& component : simulation.gainMedium->components)
                for(auto cell : domainCells(assembly, *component->domain))
                    active[cell] = 1u;

            auto const& state = *simulation.excitationState;
            if(state.values.offsets.size() != state.domains.size() + 1u || state.values.offsets.front() != 0u
               || state.values.offsets.back() != state.values.values.size())
                invalid("excitation value offsets do not match its domains");
            std::vector<double> result(assembly.cellTypes.size(), 0.0);
            std::vector<std::uint8_t> covered(assembly.cellTypes.size(), 0u);
            for(std::size_t domainIndex = 0u; domainIndex < state.domains.size(); ++domainIndex)
            {
                auto const cells = domainCells(assembly, *state.domains[domainIndex]);
                auto const begin = state.values.offsets[domainIndex];
                auto const end = state.values.offsets[domainIndex + 1u];
                auto const count = end - begin;
                if(count != 1u && count != cells.size())
                    invalid("excitation values must be scalar or match their domain size");
                for(std::size_t index = 0u; index < cells.size(); ++index)
                {
                    auto const cell = cells[index];
                    if(!active[cell])
                        invalid("excitation domain extends outside the gain medium");
                    if(covered[cell])
                        invalid("excitation domains overlap");
                    auto const value = state.values.values[begin + (count == 1u ? 0u : index)];
                    if(!std::isfinite(value) || value < 0.0 || value > 1.0)
                        invalid("excitation values must be finite and within [0, 1]");
                    result[cell] = value;
                    covered[cell] = 1u;
                }
            }
            for(std::size_t cell = 0u; cell < active.size(); ++cell)
                if(active[cell] && !covered[cell])
                    invalid("excitation domains do not cover the gain medium");
            return result;
        }

        double interpolate(std::vector<double> const& coordinates, std::vector<double> const& values, double query)
        {
            if(coordinates.empty() || coordinates.size() != values.size())
                invalid("material cross-section table is empty or inconsistent");
            if(coordinates.size() == 1u || query <= coordinates.front())
                return values.front();
            if(query >= coordinates.back())
                return values.back();
            auto const upper = std::upper_bound(coordinates.begin(), coordinates.end(), query);
            auto const index = static_cast<std::size_t>(upper - coordinates.begin());
            auto const fraction = (query - coordinates[index - 1u]) / (coordinates[index] - coordinates[index - 1u]);
            return values[index - 1u] + fraction * (values[index] - values[index - 1u]);
        }

        std::shared_ptr<Material> activeMaterial(Simulation const& simulation)
        {
            if(!simulation.gainMedium || simulation.gainMedium->components.empty())
                invalid("simulation requires a non-empty gain medium");
            auto material = simulation.gainMedium->components.front()->material;
            if(!material)
                invalid("gain component has no material");
            for(auto const& component : simulation.gainMedium->components)
                if(component->material.get() != material.get())
                    invalid("legacy backend requires gain components to share one Material object");
            if(!material->active || material->activeIonDensity <= 0.0 || !material->fluorescenceLifetime
               || !material->crossSections)
                invalid("gain material lacks active-ion density, fluorescence lifetime, or cross sections");
            if(material->bulkAttenuation && *material->bulkAttenuation != 0.0)
                invalid("legacy backend does not support bulk attenuation in active cells");
            return material;
        }

        core::PumpProfileParameters pumpProfile(PumpProfile const& profile)
        {
            core::PumpProfileParameters result;
            std::visit(
                [&result](auto const& value)
                {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr(std::is_same_v<T, UniformPumpProfile>)
                        result.kind = 0u;
                    else
                    {
                        result.kind = 1u;
                        result.radiusU = value.radiusU;
                        result.radiusV = value.radiusV;
                        result.exponent = value.exponent;
                        requireSize(value.center.values, coordinates, "pump profile center");
                        requireSize(value.axisU.values, coordinates, "pump profile axisU");
                        requireSize(value.axisV.values, coordinates, "pump profile axisV");
                        for(std::size_t coordinate = 0u; coordinate < coordinates; ++coordinate)
                        {
                            result.center[coordinate] = value.center.values[coordinate];
                            result.axisU[coordinate] = value.axisU.values[coordinate];
                            result.axisV[coordinate] = value.axisV.values[coordinate];
                        }
                    }
                },
                profile.value);
            return result;
        }
    } // namespace

    core::SimulationContext LegacyBackendConverter::convert(Simulation const& simulation)
    {
        if(simulation.opticalComponents.empty() || !simulation.phiAse || !simulation.timeIntegrationSolver)
            invalid("simulation is missing required primitive references");

        Assembly assembly;
        for(auto const& component : simulation.opticalComponents)
        {
            if(!component || !component->domain || !component->material)
                invalid("optical component is incomplete");
            selectDomain(assembly, *component->domain, true);
        }
        buildAssemblyArrays(assembly);
        auto material = activeMaterial(simulation);
        auto betaVolume = excitation(simulation, assembly);

        std::unordered_set<OpticalComponent const*> gainComponents;
        for(auto const& component : simulation.gainMedium->components)
            gainComponents.insert(component.get());
        std::vector<unsigned> claddingCellTypes(assembly.cellTypes.size(), 0u);
        std::optional<double> passiveAttenuation;
        for(auto const& component : simulation.opticalComponents)
        {
            if(gainComponents.contains(component.get()))
                continue;
            auto const attenuation = component->material->bulkAttenuation.value_or(0.0) * 1.0e-2;
            if(component->material->active || component->material->activeIonDensity != 0.0)
                invalid("components outside the gain medium must use passive materials");
            if(passiveAttenuation && *passiveAttenuation != attenuation)
                invalid("legacy backend supports one passive bulk attenuation value");
            passiveAttenuation = attenuation;
            for(auto cell : domainCells(assembly, *component->domain))
                claddingCellTypes[cell] = 1u;
        }

        SurfaceDomains surfaceDomains(assembly);

        struct OpticsEntry
        {
            int identifier;
            std::shared_ptr<SurfaceOptics> optics;
        };

        std::vector<OpticsEntry> optics;
        for(auto const& component : simulation.opticalComponents)
            for(auto const& assignment : component->surfaceOptics)
            {
                auto const identifier = surfaceDomains.assign(assignment->domain);
                auto const existing = std::find_if(
                    optics.begin(),
                    optics.end(),
                    [identifier](auto const& value) { return value.identifier == identifier; });
                if(existing != optics.end()
                   && (existing->optics->reflectivity != assignment->optics->reflectivity
                       || existing->optics->nInside != assignment->optics->nInside
                       || existing->optics->nOutside != assignment->optics->nOutside))
                    invalid("one surface domain has conflicting optical properties");
                if(existing == optics.end())
                    optics.push_back({identifier, assignment->optics});
            }

        core::SimulationRunControl run;
        run.timeStep = simulation.timeStep;
        if(!simulation.simulationSteps)
            invalid("simulation has no requested step count");
        run.numberOfSteps = narrow<unsigned>(*simulation.simulationSteps, "simulationSteps");
        run.firstSimulationStep = narrow<unsigned>(simulation.currentStep, "currentStep");
        run.aseSteps = narrow<unsigned>(simulation.phiAse->aseSteps.value_or(0u), "aseSteps");
        run.prePump = simulation.prePump;
        run.executionMode = simulation.executionMode;
        if(simulation.outputSteps)
            for(auto step : simulation.outputSteps->values)
                run.outputSteps.push_back(narrow<unsigned>(step, "outputSteps"));
        run.outputFields = simulation.outputFields;
        run.controlFields = simulation.controlFields;
        run.timeIntegration.method = simulation.timeIntegrationSolver->name;
        run.timeIntegration.implicitIterations
            = narrow<unsigned>(simulation.timeIntegrationSolver->iterations.value_or(8u), "timeIntegrator.iterations");
        run.timeIntegration.implicitTolerance = simulation.timeIntegrationSolver->tolerance.value_or(1.0e-10);

        auto const& table = *material->crossSections;
        auto const wavelengths = table.wavelengths.values;
        auto sigmaA = table.absorption.values;
        auto sigmaE = table.emission.values;
        if(wavelengths.empty() || wavelengths.size() != sigmaA.size() || wavelengths.size() != sigmaE.size())
            invalid("gain material cross-section table is empty or inconsistent");
        for(auto& value : sigmaA)
            value *= 1.0e4;
        for(auto& value : sigmaE)
            value *= 1.0e4;

        for(auto const& registration : simulation.pumpRegistrations)
        {
            if(!registration || !registration->pump || !registration->injectionMethod)
                invalid("pump registration is incomplete");
            auto const& pump = *registration->pump;
            if(!pump.spectrum || !pump.profile || !pump.angularDistribution)
                invalid("pump is missing its spectrum, profile, or angular distribution");
            core::PumpSourceParameters source;
            source.totalPower = pump.totalPower;
            source.rayCount = narrow<unsigned>(pump.rayCount, "pump.rayCount");
            source.pumpSteps = narrow<unsigned>(pump.pumpSteps.value_or(0u), "pump.pumpSteps");
            source.rngSeed = narrow<std::uint32_t>(pump.rngSeed, "pump.rngSeed");
            for(auto const& domain : registration->injectionMethod->surfaceDomains)
                source.surfaces.push_back(surfaceDomains.assign(domain));
            source.wavelengths = pump.spectrum->wavelengths.values;
            source.spectralWeights = pump.spectrum->weights.values;
            for(auto wavelength : source.wavelengths)
            {
                source.sigmaAbsorption.push_back(interpolate(wavelengths, sigmaA, wavelength));
                source.sigmaEmission.push_back(interpolate(wavelengths, sigmaE, wavelength));
            }
            source.polarAngles = pump.angularDistribution->polarAngles.values;
            source.azimuthalAngles = pump.angularDistribution->azimuthalAngles.values;
            source.angularWeights = pump.angularDistribution->weights.values;
            source.profile = pumpProfile(*pump.profile);
            for(auto const& relayValue : registration->relays)
            {
                core::PumpRelayParameters relay;
                for(auto const& domain : relayValue->exitDomains)
                    relay.exitSurfaces.push_back(surfaceDomains.assign(domain));
                for(auto const& domain : relayValue->entryDomains)
                    relay.entrySurfaces.push_back(surfaceDomains.assign(domain));
                relay.flipU = relayValue->flipU;
                relay.flipV = relayValue->flipV;
                relay.rotation = relayValue->rotation;
                requireSize(relayValue->offset.values, 2u, "pump relay offset");
                requireSize(relayValue->tilt.values, 2u, "pump relay tilt");
                relay.offset[0] = relayValue->offset.values[0];
                relay.offset[1] = relayValue->offset.values[1];
                relay.tilt[0] = relayValue->tilt.values[0];
                relay.tilt[1] = relayValue->tilt.values[1];
                relay.magnification = relayValue->magnification;
                relay.transmission = relayValue->transmission;
                source.relays.push_back(std::move(relay));
            }
            run.pump.sources.push_back(std::move(source));
        }

        int maximumSurface = 0;
        for(auto boundary : assembly.faceBoundaries)
            maximumSurface = std::max(maximumSurface, boundary);
        auto const surfaceCount = maximumSurface == 0 ? 0u : static_cast<std::size_t>(maximumSurface + 1);
        std::vector<float> surfaceReflectivity(surfaceCount, 0.0f);
        std::vector<float> surfaceInside(surfaceCount, 1.0f);
        std::vector<float> surfaceOutside(surfaceCount, 1.0f);
        for(auto const& entry : optics)
        {
            surfaceReflectivity.at(entry.identifier) = static_cast<float>(entry.optics->reflectivity);
            surfaceInside.at(entry.identifier) = static_cast<float>(entry.optics->nInside);
            surfaceOutside.at(entry.identifier) = static_cast<float>(entry.optics->nOutside);
        }

        auto const numberOfCells = assembly.cellTypes.size();
        core::HostMesh mesh(
            std::move(assembly.cellPointIndices),
            std::move(assembly.cellTypes),
            std::move(assembly.cellFaces),
            std::move(assembly.neighborCells),
            std::move(assembly.neighborLocalFaces),
            std::move(assembly.faceBoundaries),
            std::move(assembly.cellVolumes),
            std::move(assembly.points),
            std::move(assembly.samplePoints),
            std::move(assembly.cellCenters),
            std::move(betaVolume),
            std::move(claddingCellTypes),
            std::vector<float>(4u, 1.0f),
            std::vector<float>(2u * numberOfCells, 0.0f),
            std::move(surfaceReflectivity),
            std::move(surfaceInside),
            std::move(surfaceOutside),
            static_cast<float>(material->activeIonDensity * 1.0e-6),
            static_cast<float>(*material->fluorescenceLifetime),
            1u,
            passiveAttenuation.value_or(0.0));
        mesh.resultAtVolumes = true;

        core::ExperimentParameters experiment(
            narrow<unsigned>(simulation.phiAse->minRays, "phiAse.minRays"),
            narrow<unsigned>(simulation.phiAse->maxRays, "phiAse.maxRays"),
            wavelengths,
            wavelengths,
            std::move(sigmaA),
            std::move(sigmaE),
            0.0,
            0.0,
            simulation.phiAse->relativeStandardErrorThreshold,
            simulation.phiAse->useReflections,
            narrow<unsigned>(table.wavelengths.values.size(), "cross-section sample count"),
            simulation.phiAse->monochromatic);
        experiment.maxSigmaA = *std::max_element(experiment.sigmaA.begin(), experiment.sigmaA.end());
        experiment.maxSigmaE = *std::max_element(experiment.sigmaE.begin(), experiment.sigmaE.end());
        experiment.propagationMode = simulation.phiAse->propagationMode;
        experiment.forwardRayCount
            = narrow<unsigned>(simulation.phiAse->forwardRayCount.value_or(0u), "phiAse.forwardRayCount");
        experiment.reflectionMaxIterations
            = narrow<unsigned>(simulation.phiAse->reflectionMaxIterations, "phiAse.reflectionMaxIterations");
        experiment.reflectionTolerance = simulation.phiAse->reflectionTolerance;
        experiment.surfaceReservoirSize
            = narrow<unsigned>(simulation.phiAse->surfaceReservoirSize, "phiAse.surfaceReservoirSize");

        if(!simulation.phiAse->backend)
            invalid("phiAse backend was not resolved before transport");
        core::ComputeParameters compute(
            narrow<unsigned>(simulation.phiAse->repetitions, "phiAse.repetitions"),
            narrow<unsigned>(simulation.phiAse->adaptiveSteps, "phiAse.adaptiveSteps"),
            narrow<unsigned>(simulation.phiAse->numDevices, "phiAse.numDevices"),
            0u,
            *simulation.phiAse->backend,
            simulation.phiAse->parallelMode,
            simulation.phiAse->writeVtk,
            [&simulation]
            {
                std::vector<unsigned> devices;
                if(!simulation.phiAse->devices)
                    return devices;
                devices.reserve(simulation.phiAse->devices->values.size());
                for(auto const device : simulation.phiAse->devices->values)
                    devices.push_back(narrow<unsigned>(device, "phiAse.devices"));
                return devices;
            }(),
            narrow<unsigned>(simulation.phiAse->minSampleRange.value_or(0u), "phiAse.minSampleRange"),
            narrow<unsigned>(
                simulation.phiAse->maxSampleRange.value_or(mesh.numberOfCells - 1u),
                "phiAse.maxSampleRange"),
            narrow<unsigned>(
                simulation.phiAse->rngSeed.value_or(core::ComputeParameters::unspecifiedRngSeed),
                "phiAse.rngSeed"));

        core::Result result(
            std::vector<float>(mesh.numberOfCells, 0.0f),
            std::vector<double>(mesh.numberOfCells, 0.0),
            std::vector<double>(mesh.numberOfCells, 0.0),
            std::vector<unsigned>(mesh.numberOfCells, 0u),
            std::vector<double>(mesh.numberOfCells, 0.0));
        return {std::move(experiment), std::move(compute), std::move(mesh), std::move(result), std::move(run)};
    }
} // namespace hase::backend::legacy
