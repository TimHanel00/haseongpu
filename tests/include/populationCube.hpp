#pragma once

#include <data/AseDomainGraph.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <numeric>

namespace hase::test
{
    /** Unit cube, six conforming tetrahedra per voxel. No external fixtures or regenerated references. */
    inline data::TraceData populationCube(std::uint32_t const subdivisions, float const reflectivity)
    {
        data::TraceData mesh;
        auto const width = subdivisions + 1u;
        mesh.numberOfMeshPoints = width * width * width;
        mesh.numberOfMaterials = 1u;
        auto id = [width](std::array<std::uint32_t, 3u> const p) { return p[0u] + width * (p[1u] + width * p[2u]); };
        mesh.points.resize(3u * mesh.numberOfMeshPoints);
        for(std::uint32_t z = 0u; z < width; ++z)
            for(std::uint32_t y = 0u; y < width; ++y)
                for(std::uint32_t x = 0u; x < width; ++x)
                {
                    auto const vertex = id({x, y, z});
                    mesh.points[vertex] = static_cast<double>(x) / subdivisions;
                    mesh.points[mesh.numberOfMeshPoints + vertex] = static_cast<double>(y) / subdivisions;
                    mesh.points[2u * mesh.numberOfMeshPoints + vertex] = static_cast<double>(z) / subdivisions;
                }
        for(std::uint32_t z = 0u; z < subdivisions; ++z)
            for(std::uint32_t y = 0u; y < subdivisions; ++y)
                for(std::uint32_t x = 0u; x < subdivisions; ++x)
                {
                    std::array<std::uint32_t, 3u> permutation{0u, 1u, 2u};
                    do
                    {
                        std::array<std::uint32_t, 3u> corner{x, y, z};
                        mesh.cellPointIndices.push_back(id(corner));
                        for(auto const axis : permutation)
                        {
                            ++corner[axis];
                            mesh.cellPointIndices.push_back(id(corner));
                        }
                    } while(std::next_permutation(permutation.begin(), permutation.end()));
                }
        mesh.numberOfCells = static_cast<std::uint32_t>(mesh.cellPointIndices.size() / 4u);
        mesh.cellNeighborCells.assign(4u * mesh.numberOfCells, -1);
        mesh.cellNeighborLocalFaces.assign(4u * mesh.numberOfCells, -1);
        mesh.cellFaceBoundaries.assign(4u * mesh.numberOfCells, 1);
        mesh.cellCenters.assign(3u * mesh.numberOfCells, 0.0);
        std::map<std::array<std::uint32_t, 3u>, std::pair<std::uint32_t, std::uint32_t>> faces;
        for(std::uint32_t cell = 0u; cell < mesh.numberOfCells; ++cell)
        {
            for(std::uint32_t vertex = 0u; vertex < 4u; ++vertex)
                for(std::uint32_t axis = 0u; axis < 3u; ++axis)
                    mesh.cellCenters[axis * mesh.numberOfCells + cell]
                        += mesh.points[axis * mesh.numberOfMeshPoints + mesh.cellPointIndices[4u * cell + vertex]]
                           / 4.0;
            for(std::uint32_t face = 0u; face < 4u; ++face)
            {
                std::array<std::uint32_t, 3u> key{};
                std::uint32_t next = 0u;
                for(std::uint32_t vertex = 0u; vertex < 4u; ++vertex)
                    if(vertex != face)
                    {
                        key[next++] = mesh.cellPointIndices[4u * cell + vertex];
                        mesh.cellFaces.push_back(static_cast<int>(mesh.cellPointIndices[4u * cell + vertex]));
                    }
                std::sort(key.begin(), key.end());
                auto const [found, inserted] = faces.emplace(key, std::pair{cell, face});
                if(!inserted)
                {
                    auto const [otherCell, otherFace] = found->second;
                    mesh.cellNeighborCells[4u * cell + face] = static_cast<int>(otherCell);
                    mesh.cellNeighborLocalFaces[4u * cell + face] = static_cast<int>(otherFace);
                    mesh.cellNeighborCells[4u * otherCell + otherFace] = static_cast<int>(cell);
                    mesh.cellNeighborLocalFaces[4u * otherCell + otherFace] = static_cast<int>(face);
                    mesh.cellFaceBoundaries[4u * cell + face] = -1;
                    mesh.cellFaceBoundaries[4u * otherCell + otherFace] = -1;
                }
            }
        }
        mesh.cellTypes.assign(mesh.numberOfCells, data::vtkTetraCellType);
        mesh.cellVolumes.assign(mesh.numberOfCells, 1.0 / mesh.numberOfCells);
        mesh.betaVolume.assign(mesh.numberOfCells, 0.5);
        mesh.cellMaterialIds.assign(mesh.numberOfCells, 0u);
        mesh.materialActive = {1u};
        mesh.materialRefractiveIndices = {1.0};
        mesh.materialActiveIonDensities = {1.0};
        mesh.materialFluorescenceLifetimes = {1.0};
        mesh.materialBulkAttenuations = {0.0};
        mesh.materialPeakAbsorption = {0.0};
        mesh.materialPeakEmission = {0.0};
        mesh.materialCrossSectionOffsets = {0u, 2u};
        mesh.crossSectionWavelengths = {1.0, 2.0};
        mesh.crossSectionAbsorption = {0.0, 0.0};
        mesh.crossSectionEmission = {0.0, 0.0};
        mesh.surfaceReflectivities.assign(4u * mesh.numberOfCells, reflectivity);
        mesh.surfaceRefractiveIndexInside.assign(4u * mesh.numberOfCells, 1.0f);
        mesh.surfaceRefractiveIndexOutside.assign(4u * mesh.numberOfCells, 1.0f);
        mesh.rebuildStaticPrefixes();
        mesh.precomputeBarycentricFacePlanes();
        return mesh;
    }

    inline data::AseDomainGraph populationSources(data::TraceData const& mesh)
    {
        data::AseDomainGraph graph;
        graph.domainCellOffsets = {0u, mesh.numberOfCells};
        graph.domainGlobalCells.resize(mesh.numberOfCells);
        std::iota(graph.domainGlobalCells.begin(), graph.domainGlobalCells.end(), 0u);
        graph.domainSourceStrengthPrefix = mesh.sourceStrengthPrefix;
        graph.domainSourceStrengthTotals = {mesh.sourceStrengthPrefix.back()};
        return graph;
    }
} // namespace hase::test
