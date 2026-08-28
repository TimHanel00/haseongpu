#include <data/SimulationPreparation.hpp>
#include <data/TraceData.hpp>

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

namespace hase::internal::simulationPreparation
{
    using namespace hase::data;
    using hase::data::ExcitationLayout;

    constexpr std::size_t coordinates = 3u;
    constexpr std::size_t verticesPerCell = 4u;
    constexpr std::size_t facesPerCell = 4u;
    constexpr std::size_t verticesPerFace = 3u;

    [[noreturn]] void invalid(std::string const& message)
    {
        throw std::runtime_error("simulation preparation: " + message);
    }

    template<typename T>
    T narrow(std::uint64_t value, std::string const& name)
    {
        if(value > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
            invalid(name + " is outside the runtime range");
        return static_cast<T>(value);
    }

    template<typename T>
    void requireSize(std::vector<T> const& values, std::size_t expected, std::string const& name)
    {
        if(values.size() != expected)
            invalid(name + " has " + std::to_string(values.size()) + " values; expected " + std::to_string(expected));
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
        requireSize(topology.neighborLocalFaces.values, count * facesPerCell, "volume topology neighborLocalFaces");
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
        std::vector<double> cellVolumes;
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
                assembly.cellVolumes.push_back(topology.cellVolumes.values[cell]);
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

    bool domainsOverlap(Domain const& left, Domain const& right)
    {
        for(auto const& leftShard : shards(left))
            for(auto const& rightShard : shards(right))
                if(leftShard.topology.get() == rightShard.topology.get())
                {
                    requireSize(rightShard.mask, leftShard.mask.size(), "surface domain mask");
                    for(std::size_t index = 0u; index < leftShard.mask.size(); ++index)
                        if(leftShard.mask[index] && rightShard.mask[index])
                            return true;
                }
        return false;
    }

    ExcitationLayout excitationUpdatePlan(Simulation const& simulation, Assembly const& assembly)
    {
        if(!simulation.excitationState || !simulation.gainMedium)
            invalid("simulation has no excitation state or gain medium");
        std::vector<std::uint8_t> active(assembly.cellTypes.size(), 0u);
        for(auto const& component : simulation.gainMedium->components)
            for(auto cell : domainCells(assembly, *component->domain))
                active[cell] = 1u;
        std::vector<std::vector<unsigned>> domainCellIndices;
        domainCellIndices.reserve(simulation.excitationState->domains.size());
        for(auto const& domain : simulation.excitationState->domains)
            domainCellIndices.push_back(domainCells(assembly, *domain));
        return ExcitationLayout{std::move(active), std::move(domainCellIndices)};
    }

    void validateCrossSections(CrossSectionTable const& table, std::string const& materialName)
    {
        auto const& wavelengths = table.wavelengths.values;
        if(wavelengths.empty() || wavelengths.size() != table.absorption.values.size()
           || wavelengths.size() != table.emission.values.size())
            invalid("material '" + materialName + "' has an empty or inconsistent cross-section table");
        for(std::size_t sample = 0u; sample < wavelengths.size(); ++sample)
        {
            if(!std::isfinite(wavelengths[sample]) || wavelengths[sample] <= 0.0
               || !std::isfinite(table.absorption.values[sample]) || !std::isfinite(table.emission.values[sample])
               || table.absorption.values[sample] < 0.0 || table.emission.values[sample] < 0.0)
                invalid("material '" + materialName + "' has invalid cross-section samples");
        }
        bool const monochromatic
            = std::ranges::all_of(wavelengths, [&](double wavelength) { return wavelength == wavelengths.front(); });
        if(!monochromatic)
            for(std::size_t sample = 1u; sample < wavelengths.size(); ++sample)
                if(wavelengths[sample] <= wavelengths[sample - 1u])
                    invalid("material '" + materialName + "' wavelengths must be strictly increasing");
    }

    struct MaterialAssembly
    {
        std::vector<unsigned> cellMaterialIds;
        std::vector<std::uint8_t> active;
        std::vector<double> refractiveIndices;
        std::vector<double> activeIonDensities;
        std::vector<double> fluorescenceLifetimes;
        std::vector<double> bulkAttenuations;
        std::vector<double> peakAbsorption;
        std::vector<double> peakEmission;
        std::vector<unsigned> crossSectionOffsets{0u};
        std::vector<double> wavelengths;
        std::vector<double> absorption;
        std::vector<double> emission;
    };

    MaterialAssembly assembleMaterials(Simulation const& simulation, Assembly const& assembly)
    {
        MaterialAssembly result;
        result.cellMaterialIds.assign(assembly.cellTypes.size(), std::numeric_limits<unsigned>::max());
        std::unordered_map<Material const*, unsigned> materialIds;

        for(auto const& component : simulation.opticalComponents)
        {
            auto const& material = component->material;
            auto [position, inserted] = materialIds.emplace(material.get(), static_cast<unsigned>(materialIds.size()));
            unsigned const materialId = position->second;
            if(inserted)
            {
                if(!std::isfinite(material->refractiveIndex) || material->refractiveIndex <= 0.0)
                    invalid("material '" + material->materialName + "' has an invalid refractive index");
                if(!std::isfinite(material->activeIonDensity) || material->activeIonDensity < 0.0)
                    invalid("material '" + material->materialName + "' has an invalid active-ion density");
                double const lifetime = material->fluorescenceLifetime.value_or(0.0);
                if(material->active
                   && (material->activeIonDensity <= 0.0 || lifetime <= 0.0 || !material->crossSections))
                    invalid(
                        "active material '" + material->materialName
                        + "' requires positive active-ion density, fluorescence lifetime, and cross sections");
                double const attenuation = material->bulkAttenuation.value_or(0.0);
                if(!std::isfinite(attenuation) || attenuation < 0.0)
                    invalid("material '" + material->materialName + "' has invalid bulk attenuation");

                result.active.push_back(material->active ? 1u : 0u);
                result.refractiveIndices.push_back(material->refractiveIndex);
                result.activeIonDensities.push_back(material->activeIonDensity);
                result.fluorescenceLifetimes.push_back(lifetime);
                result.bulkAttenuations.push_back(attenuation);

                double peakAbsorption = 0.0;
                double peakEmission = 0.0;
                if(material->crossSections)
                {
                    validateCrossSections(*material->crossSections, material->materialName);
                    auto const& table = *material->crossSections;
                    auto const peak = std::max_element(table.emission.values.begin(), table.emission.values.end());
                    auto const peakIndex = static_cast<std::size_t>(peak - table.emission.values.begin());
                    peakAbsorption = table.absorption.values[peakIndex];
                    peakEmission = table.emission.values[peakIndex];
                    result.wavelengths.insert(
                        result.wavelengths.end(),
                        table.wavelengths.values.begin(),
                        table.wavelengths.values.end());
                    result.absorption.insert(
                        result.absorption.end(),
                        table.absorption.values.begin(),
                        table.absorption.values.end());
                    result.emission.insert(
                        result.emission.end(),
                        table.emission.values.begin(),
                        table.emission.values.end());
                }
                result.peakAbsorption.push_back(peakAbsorption);
                result.peakEmission.push_back(peakEmission);
                result.crossSectionOffsets.push_back(static_cast<unsigned>(result.wavelengths.size()));
            }

            for(auto const cell : domainCells(assembly, *component->domain))
            {
                if(result.cellMaterialIds[cell] != std::numeric_limits<unsigned>::max())
                    invalid("optical component domains overlap");
                result.cellMaterialIds[cell] = materialId;
            }
        }

        if(std::ranges::find(result.cellMaterialIds, std::numeric_limits<unsigned>::max())
           != result.cellMaterialIds.end())
            invalid("optical components do not assign a material to every trace cell");
        return result;
    }

    TraceData extractAseDomainTrace(
        TraceData const& global,
        std::vector<unsigned> const& globalCells,
        std::vector<std::uint32_t>& localToGlobalPoints)
    {
        std::vector<int> globalToLocalCell(global.numberOfCells, -1);
        for(std::size_t local = 0u; local < globalCells.size(); ++local)
            globalToLocalCell.at(globalCells[local]) = static_cast<int>(local);

        std::vector<int> globalToLocalPoint(global.numberOfMeshPoints, -1);
        for(auto const globalCell : globalCells)
            for(std::size_t vertex = 0u; vertex < global.numberOfCellVertices; ++vertex)
            {
                auto const globalPoint = global.cellPointIndices.at(globalCell * global.numberOfCellVertices + vertex);
                if(globalToLocalPoint.at(globalPoint) < 0)
                {
                    globalToLocalPoint[globalPoint] = static_cast<int>(localToGlobalPoints.size());
                    localToGlobalPoints.push_back(static_cast<std::uint32_t>(globalPoint));
                }
            }

        std::vector<double> points(3u * localToGlobalPoints.size());
        for(std::size_t local = 0u; local < localToGlobalPoints.size(); ++local)
            for(std::size_t coordinate = 0u; coordinate < 3u; ++coordinate)
                points[coordinate * localToGlobalPoints.size() + local]
                    = global.points[coordinate * global.numberOfMeshPoints + localToGlobalPoints[local]];

        std::vector<unsigned> cellPointIndices;
        std::vector<unsigned> cellTypes;
        std::vector<int> cellFaces;
        std::vector<int> neighborCells;
        std::vector<int> neighborLocalFaces;
        std::vector<int> faceBoundaries;
        std::vector<double> cellVolumes;
        std::vector<double> cellCenters(3u * globalCells.size());
        std::vector<double> betaVolume;
        std::vector<unsigned> cellMaterialIds;
        cellPointIndices.reserve(globalCells.size() * global.numberOfCellVertices);
        cellTypes.reserve(globalCells.size());
        cellFaces.reserve(globalCells.size() * global.numberOfFacesPerCell * data::tet4FaceWidth);
        neighborCells.reserve(globalCells.size() * global.numberOfFacesPerCell);
        neighborLocalFaces.reserve(globalCells.size() * global.numberOfFacesPerCell);
        faceBoundaries.reserve(globalCells.size() * global.numberOfFacesPerCell);
        cellVolumes.reserve(globalCells.size());
        betaVolume.reserve(globalCells.size());
        cellMaterialIds.reserve(globalCells.size());

        for(std::size_t localCell = 0u; localCell < globalCells.size(); ++localCell)
        {
            auto const globalCell = globalCells[localCell];
            cellTypes.push_back(global.cellTypes.at(globalCell));
            cellVolumes.push_back(global.cellVolumes.at(globalCell));
            betaVolume.push_back(global.betaVolume.at(globalCell));
            cellMaterialIds.push_back(global.cellMaterialIds.at(globalCell));
            for(std::size_t coordinate = 0u; coordinate < 3u; ++coordinate)
                cellCenters[coordinate * globalCells.size() + localCell]
                    = global.cellCenters[coordinate * global.numberOfCells + globalCell];
            for(std::size_t vertex = 0u; vertex < global.numberOfCellVertices; ++vertex)
            {
                auto const globalPoint = global.cellPointIndices.at(globalCell * global.numberOfCellVertices + vertex);
                cellPointIndices.push_back(static_cast<unsigned>(globalToLocalPoint.at(globalPoint)));
            }
            for(std::size_t face = 0u; face < global.numberOfFacesPerCell; ++face)
            {
                auto const globalFace = globalCell * global.numberOfFacesPerCell + face;
                for(std::size_t vertex = 0u; vertex < data::tet4FaceWidth; ++vertex)
                {
                    auto const globalPoint = global.cellFaces.at(globalFace * data::tet4FaceWidth + vertex);
                    cellFaces.push_back(globalToLocalPoint.at(static_cast<std::size_t>(globalPoint)));
                }
                auto const globalNeighbor = global.cellNeighborCells.at(globalFace);
                auto const localNeighbor
                    = globalNeighbor < 0 ? -1 : globalToLocalCell.at(static_cast<std::size_t>(globalNeighbor));
                neighborCells.push_back(localNeighbor);
                neighborLocalFaces.push_back(localNeighbor < 0 ? -1 : global.cellNeighborLocalFaces.at(globalFace));
                faceBoundaries.push_back(global.cellFaceBoundaries.at(globalFace));
            }
        }

        auto samplePoints = cellCenters;
        return TraceData(
            std::move(cellPointIndices),
            std::move(cellTypes),
            std::move(cellFaces),
            std::move(neighborCells),
            std::move(neighborLocalFaces),
            std::move(faceBoundaries),
            std::move(cellVolumes),
            std::move(points),
            std::move(samplePoints),
            std::move(cellCenters),
            std::move(betaVolume),
            std::move(cellMaterialIds),
            global.materialActive,
            global.materialRefractiveIndices,
            global.materialActiveIonDensities,
            global.materialFluorescenceLifetimes,
            global.materialBulkAttenuations,
            global.materialPeakAbsorption,
            global.materialPeakEmission,
            global.materialCrossSectionOffsets,
            global.crossSectionWavelengths,
            global.crossSectionAbsorption,
            global.crossSectionEmission,
            global.surfaceReflectivities,
            global.surfaceRefractiveIndexInside,
            global.surfaceRefractiveIndexOutside);
    }

    AseDomainGraph prepareAseDomainGraph(
        Simulation const& simulation,
        Assembly const& assembly,
        TraceData const& global)
    {
        AseDomainGraph graph;
        graph.globalCellDomains.assign(global.numberOfCells, std::numeric_limits<DomainId>::max());
        auto const globalFaceCount = global.numberOfCells * global.numberOfFacesPerCell;
        graph.globalBoundaryTargetDomains.assign(globalFaceCount, invalidDomainId);
        graph.globalBoundaryTargetCells.assign(globalFaceCount, std::numeric_limits<std::uint32_t>::max());
        graph.globalBoundaryTargetFaces.assign(globalFaceCount, std::numeric_limits<std::uint32_t>::max());
        graph.globalBoundaryReflectivities.assign(globalFaceCount, 0.0f);
        graph.globalBoundarySourceRefractiveIndices.assign(globalFaceCount, 1.0f);
        graph.globalBoundaryTargetRefractiveIndices.assign(globalFaceCount, 1.0f);
        graph.domains.reserve(simulation.opticalComponents.size());
        std::vector<std::vector<int>> globalToLocalCells;
        for(std::size_t index = 0u; index < simulation.opticalComponents.size(); ++index)
        {
            auto cells = domainCells(assembly, *simulation.opticalComponents[index]->domain);
            std::ranges::sort(cells);
            std::vector<std::uint32_t> localToGlobalCells(cells.begin(), cells.end());
            std::vector<std::uint32_t> localToGlobalPoints;
            auto trace = extractAseDomainTrace(global, cells, localToGlobalPoints);
            auto const id = static_cast<DomainId>(index);
            std::vector<int> inverse(global.numberOfCells, -1);
            for(std::size_t local = 0u; local < cells.size(); ++local)
            {
                graph.globalCellDomains.at(cells[local]) = id;
                inverse[cells[local]] = static_cast<int>(local);
            }
            globalToLocalCells.push_back(std::move(inverse));
            graph.domains.push_back(
                {id,
                 std::move(trace),
                 std::move(localToGlobalCells),
                 std::move(localToGlobalPoints),
                 std::vector<DomainId>(cells.size(), id),
                 std::vector<DomainId>(cells.size() * global.numberOfFacesPerCell, invalidDomainId),
                 std::vector<std::uint32_t>(
                     cells.size() * global.numberOfFacesPerCell,
                     std::numeric_limits<std::uint32_t>::max()),
                 std::vector<std::uint32_t>(
                     cells.size() * global.numberOfFacesPerCell,
                     std::numeric_limits<std::uint32_t>::max()),
                 std::vector<float>(cells.size() * global.numberOfFacesPerCell, 0.0f),
                 std::vector<float>(cells.size() * global.numberOfFacesPerCell, 1.0f),
                 std::vector<float>(cells.size() * global.numberOfFacesPerCell, 1.0f),
                 simulation.opticalComponents[index]->aseRays});
        }

        graph.domainCellOffsets.push_back(0u);
        double globalSourceStrength = 0.0;
        for(auto const& domain : graph.domains)
        {
            graph.domainGlobalCells.insert(
                graph.domainGlobalCells.end(),
                domain.localToGlobalCells.begin(),
                domain.localToGlobalCells.end());
            double previousLocalStrength = 0.0;
            for(double const localPrefix : domain.trace.sourceStrengthPrefix)
            {
                globalSourceStrength += localPrefix - previousLocalStrength;
                previousLocalStrength = localPrefix;
                graph.domainSourceStrengthPrefix.push_back(globalSourceStrength);
            }
            graph.domainSourceStrengthTotals.push_back(previousLocalStrength);
            graph.domainCellOffsets.push_back(static_cast<std::uint32_t>(graph.domainGlobalCells.size()));
        }

        for(auto& domain : graph.domains)
            for(std::size_t localCell = 0u; localCell < domain.localToGlobalCells.size(); ++localCell)
                for(std::size_t face = 0u; face < global.numberOfFacesPerCell; ++face)
                {
                    auto const globalCell = domain.localToGlobalCells[localCell];
                    auto const globalFace = globalCell * global.numberOfFacesPerCell + face;
                    auto const neighbor = global.cellNeighborCells[globalFace];
                    if(neighbor < 0)
                        continue;
                    auto const targetDomain = graph.globalCellDomains.at(static_cast<std::size_t>(neighbor));
                    if(targetDomain == domain.id)
                        continue;
                    auto const targetLocal
                        = globalToLocalCells.at(targetDomain).at(static_cast<std::size_t>(neighbor));
                    if(targetLocal < 0)
                        invalid("inter-domain neighbor is absent from its component-local trace");

                    auto const oppositeFace = global.cellNeighborLocalFaces.at(globalFace);
                    if(oppositeFace < 0 || static_cast<std::size_t>(oppositeFace) >= global.numberOfFacesPerCell)
                        invalid("inter-domain interface has no reciprocal local face");
                    auto const sourceBoundary = global.cellFaceBoundaries.at(globalFace);
                    auto const targetGlobalFace = static_cast<std::size_t>(neighbor) * global.numberOfFacesPerCell
                                                  + static_cast<std::size_t>(oppositeFace);
                    auto const targetBoundary = global.cellFaceBoundaries.at(targetGlobalFace);
                    auto const reflectivity = [&](int const boundary)
                    { return boundary > 0 ? static_cast<double>(global.surfaceReflectivities.at(boundary)) : 0.0; };
                    double const sourceReflectivity = reflectivity(sourceBoundary);
                    double const targetReflectivity = reflectivity(targetBoundary);
                    if(sourceBoundary > 0 && targetBoundary > 0
                       && std::abs(sourceReflectivity - targetReflectivity) > 1.0e-7)
                        invalid("the two sides of an inter-domain interface have different reflectivities");
                    double const sharedReflectivity = sourceBoundary > 0 ? sourceReflectivity : targetReflectivity;
                    auto const sourceMaterial = global.cellMaterialIds.at(globalCell);
                    auto const targetMaterial = global.cellMaterialIds.at(static_cast<std::size_t>(neighbor));
                    double const sourceIndex = global.materialRefractiveIndices.at(sourceMaterial);
                    double const targetIndex = global.materialRefractiveIndices.at(targetMaterial);

                    if(domain.trace.surfaceReflectivities.empty())
                    {
                        domain.trace.surfaceReflectivities.push_back(0.0f);
                        domain.trace.surfaceRefractiveIndexInside.push_back(1.0f);
                        domain.trace.surfaceRefractiveIndexOutside.push_back(1.0f);
                    }
                    auto const boundaryId = static_cast<std::uint32_t>(domain.trace.surfaceReflectivities.size());
                    domain.trace.surfaceReflectivities.push_back(static_cast<float>(sharedReflectivity));
                    domain.trace.surfaceRefractiveIndexInside.push_back(static_cast<float>(sourceIndex));
                    domain.trace.surfaceRefractiveIndexOutside.push_back(static_cast<float>(targetIndex));
                    domain.trace.cellFaceBoundaries.at(localCell * global.numberOfFacesPerCell + face)
                        = static_cast<int>(boundaryId);
                    auto const localFaceIndex = localCell * global.numberOfFacesPerCell + face;
                    domain.boundaryTargetDomains.at(localFaceIndex) = targetDomain;
                    domain.boundaryTargetCells.at(localFaceIndex) = static_cast<std::uint32_t>(targetLocal);
                    domain.boundaryTargetFaces.at(localFaceIndex) = static_cast<std::uint32_t>(oppositeFace);
                    domain.boundaryReflectivities.at(localFaceIndex) = static_cast<float>(sharedReflectivity);
                    domain.boundarySourceRefractiveIndices.at(localFaceIndex) = static_cast<float>(sourceIndex);
                    domain.boundaryTargetRefractiveIndices.at(localFaceIndex) = static_cast<float>(targetIndex);
                    graph.globalBoundaryTargetDomains.at(globalFace) = targetDomain;
                    graph.globalBoundaryTargetCells.at(globalFace) = static_cast<std::uint32_t>(neighbor);
                    graph.globalBoundaryTargetFaces.at(globalFace) = static_cast<std::uint32_t>(oppositeFace);
                    graph.globalBoundaryReflectivities.at(globalFace) = static_cast<float>(sharedReflectivity);
                    graph.globalBoundarySourceRefractiveIndices.at(globalFace) = static_cast<float>(sourceIndex);
                    graph.globalBoundaryTargetRefractiveIndices.at(globalFace) = static_cast<float>(targetIndex);
                    graph.interfaces.push_back(
                        {domain.id,
                         targetDomain,
                         static_cast<std::uint32_t>(localCell),
                         static_cast<std::uint32_t>(face),
                         static_cast<std::uint32_t>(targetLocal),
                         static_cast<std::uint32_t>(oppositeFace),
                         boundaryId,
                         sharedReflectivity,
                         sourceIndex,
                         targetIndex});
                }
        return graph;
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
} // namespace hase::internal::simulationPreparation

namespace hase::data
{
    using namespace hase::internal::simulationPreparation;

    ExcitationLayout::ExcitationLayout(
        std::vector<std::uint8_t> active,
        std::vector<std::vector<unsigned>> domainCells)
        : m_active(std::move(active))
        , m_domainCells(std::move(domainCells))
    {
    }

    std::vector<double> ExcitationLayout::values(ExcitationState const& state) const
    {
        if(state.values.offsets.size() != m_domainCells.size() + 1u || state.values.offsets.empty()
           || state.values.offsets.front() != 0u || state.values.offsets.back() != state.values.values.size())
            invalid("excitation value offsets do not match its domains");
        std::vector<double> result(m_active.size(), 0.0);
        std::vector<std::uint8_t> covered(m_active.size(), 0u);
        for(std::size_t domainIndex = 0u; domainIndex < m_domainCells.size(); ++domainIndex)
        {
            auto const& cells = m_domainCells[domainIndex];
            auto const begin = state.values.offsets[domainIndex];
            auto const end = state.values.offsets[domainIndex + 1u];
            auto const count = end - begin;
            if(count != 1u && count != cells.size())
                invalid("excitation values must be scalar or match their domain size");
            for(std::size_t index = 0u; index < cells.size(); ++index)
            {
                auto const cell = cells[index];
                if(!m_active[cell])
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
        for(std::size_t cell = 0u; cell < m_active.size(); ++cell)
            if(m_active[cell] && !covered[cell])
                invalid("excitation domains do not cover the gain medium");
        return result;
    }

    void ExcitationLayout::apply(ExcitationState const& state, std::vector<double>& destination) const
    {
        destination = values(state);
    }

    SimulationState prepareSimulation(Simulation const& simulation)
    {
        return std::move(prepareSimulationWithUpdates(simulation).state);
    }

    SimulationPreparation prepareSimulationWithUpdates(Simulation const& simulation)
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
        auto excitationPlan = excitationUpdatePlan(simulation, assembly);
        auto betaVolume = excitationPlan.values(*simulation.excitationState);
        auto materials = assembleMaterials(simulation, assembly);

        SurfaceDomains surfaceDomains(assembly);

        struct OpticsEntry
        {
            int identifier;
            std::shared_ptr<Domain> domain;
            std::shared_ptr<SurfaceOptics> optics;
        };

        std::vector<OpticsEntry> optics;
        for(auto const& component : simulation.opticalComponents)
            for(auto const& assignment : component->surfaceOptics)
            {
                if(!assignment || !assignment->domain || !assignment->optics)
                    invalid("surface-optics assignment is incomplete");
                auto const identifier = surfaceDomains.assign(assignment->domain);
                auto const existing = std::find_if(
                    optics.begin(),
                    optics.end(),
                    [identifier](auto const& value) { return value.identifier == identifier; });
                auto const overlap = std::find_if(
                    optics.begin(),
                    optics.end(),
                    [&](auto const& value) { return domainsOverlap(*value.domain, *assignment->domain); });
                if(overlap != optics.end() && overlap->identifier != identifier)
                    invalid("surface-optics assignments overlap on one face");
                if(existing != optics.end()
                   && (existing->optics->reflectivity != assignment->optics->reflectivity
                       || existing->optics->nInside != assignment->optics->nInside
                       || existing->optics->nOutside != assignment->optics->nOutside))
                    invalid("one surface domain has conflicting optical properties");
                if(existing == optics.end())
                    optics.push_back({identifier, assignment->domain, assignment->optics});
            }

        core::SimulationControls run;
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

        TraceData trace(
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
            std::move(materials.cellMaterialIds),
            std::move(materials.active),
            std::move(materials.refractiveIndices),
            std::move(materials.activeIonDensities),
            std::move(materials.fluorescenceLifetimes),
            std::move(materials.bulkAttenuations),
            std::move(materials.peakAbsorption),
            std::move(materials.peakEmission),
            std::move(materials.crossSectionOffsets),
            std::move(materials.wavelengths),
            std::move(materials.absorption),
            std::move(materials.emission),
            std::move(surfaceReflectivity),
            std::move(surfaceInside),
            std::move(surfaceOutside));
        auto aseDomains = prepareAseDomainGraph(simulation, assembly, trace);

        core::AseTraceControls ase;
        ase.minRays = narrow<unsigned>(simulation.phiAse->minRays, "phiAse.minRays");
        ase.maxRays = narrow<unsigned>(simulation.phiAse->maxRays, "phiAse.maxRays");
        ase.relativeStandardErrorThreshold = simulation.phiAse->relativeStandardErrorThreshold;
        ase.enableDiagnostics = simulation.phiAse->enableDiagnostics;
        ase.useReflections = simulation.phiAse->useReflections;
        ase.reflectionMode = simulation.phiAse->reflectionMode;
        ase.surfaceReservoirSize
            = narrow<unsigned>(simulation.phiAse->surfaceReservoirSize, "phiAse.surfaceReservoirSize");
        ase.srmPositionMode = simulation.phiAse->srmPositionMode;
        if(ase.reflectionMode != "direct" && ase.reflectionMode != "srm")
            invalid("phiAse.reflectionMode must be 'direct' or 'srm'");
        if(ase.surfaceReservoirSize == 0u)
            invalid("phiAse.surfaceReservoirSize must be positive");
        if(ase.srmPositionMode != "exact" && ase.srmPositionMode != "centroid")
            invalid("phiAse.srmPositionMode must be 'exact' or 'centroid'");
        ase.monochromatic = simulation.phiAse->monochromatic;
        ase.propagationMode = simulation.phiAse->propagationMode;
        ase.forwardRayCount
            = narrow<unsigned>(simulation.phiAse->forwardRayCount.value_or(0u), "phiAse.forwardRayCount");
        ase.reflectionMaxIterations
            = narrow<unsigned>(simulation.phiAse->reflectionMaxIterations, "phiAse.reflectionMaxIterations");
        ase.reflectionTolerance = simulation.phiAse->reflectionTolerance;
        ase.boundaryMaxPasses
            = narrow<unsigned>(simulation.phiAse->boundaryMaxPasses.value_or(0u), "phiAse.boundaryMaxPasses");
        ase.domainCount = aseDomains.domains.size();

        if(!simulation.phiAse->backend)
            invalid("phiAse backend was not resolved before transport");
        core::ExecutionPolicy compute(
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
                simulation.phiAse->maxSampleRange.value_or(trace.numberOfCells - 1u),
                "phiAse.maxSampleRange"),
            narrow<unsigned>(
                simulation.phiAse->rngSeed.value_or(core::ExecutionPolicy::unspecifiedRngSeed),
                "phiAse.rngSeed"));

        PhiAseResult result(
            std::vector<float>(trace.numberOfCells, 0.0f),
            std::vector<double>(trace.numberOfCells, 0.0),
            std::vector<double>(trace.numberOfCells, 0.0),
            std::vector<unsigned>(trace.numberOfCells, 0u),
            std::vector<double>(trace.numberOfCells, 0.0));
        SimulationState state{
            std::move(ase),
            std::move(compute),
            std::move(trace),
            std::move(aseDomains),
            std::move(result),
            std::move(run)};
        return {std::move(state), std::move(excitationPlan)};
    }
} // namespace hase::data
