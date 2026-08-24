/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <data/TraceData.hpp>

#include <stdexcept>
#include <vector>

namespace hase::kernels
{
    /**
     * @param mesh Prepared host trace containing material and point connectivity.
     * @param cell Cell index in prepared order.
     * @param localVertex Local vertex index within the cell.
     * @return Flat index into material-separated vertex storage.
     */
    [[nodiscard]] inline unsigned materialVertexIndex(
        data::TraceData const& mesh,
        unsigned const cell,
        unsigned const localVertex)
    {
        unsigned const point = mesh.cellPointIndices.at(cell * mesh.numberOfCellVertices + localVertex);
        return point + mesh.cellMaterialIds.at(cell) * mesh.numberOfMeshPoints;
    }

    /**
     * @param mesh Prepared host trace containing cell volumes and connectivity.
     * @return Lumped volume for every material-vertex pair. Shared geometric
     * vertices remain separated at material interfaces.
     */
    [[nodiscard]] inline std::vector<double> makeLumpedMaterialVertexVolumes(data::TraceData const& mesh)
    {
        // Materials use separate values at a shared geometric vertex so
        // accumulation cannot smear a physical discontinuity across the interface.
        std::vector result(mesh.numberOfMaterials * mesh.numberOfMeshPoints, 0.0);
        for(unsigned cell = 0u; cell < mesh.numberOfCells; ++cell)
        {
            double const share
                = static_cast<double>(mesh.cellVolumes.at(cell)) / static_cast<double>(mesh.numberOfCellVertices);
            for(unsigned localVertex = 0u; localVertex < mesh.numberOfCellVertices; ++localVertex)
                result.at(materialVertexIndex(mesh, cell, localVertex)) += share;
        }
        return result;
    }

    /**
     * @brief Convert material-vertex integrals into volume-averaged cell densities.
     * @param mesh Prepared host trace defining volumes and connectivity.
     * @param vertexIntegrals Integral for every material-vertex pair.
     * @return One density per cell in prepared order.
     * @throws std::runtime_error If the input size does not match the mesh layout.
     */
    [[nodiscard]] inline std::vector<double> accumulateMaterialVertexIntegralsToCellDensities(
        data::TraceData const& mesh,
        std::vector<double> const& vertexIntegrals)
    {
        if(vertexIntegrals.size() != mesh.numberOfMaterials * mesh.numberOfMeshPoints)
            throw std::runtime_error("material vertex integral count does not match the mesh");

        auto const lumpedVertexVolumes = makeLumpedMaterialVertexVolumes(mesh);
        std::vector result(mesh.numberOfCells, 0.0);
        for(unsigned cell = 0u; cell < mesh.numberOfCells; ++cell)
        {
            double densitySum = 0.0;
            for(unsigned localVertex = 0u; localVertex < mesh.numberOfCellVertices; ++localVertex)
            {
                unsigned const vertex = materialVertexIndex(mesh, cell, localVertex);
                double const volume = lumpedVertexVolumes.at(vertex);
                densitySum += volume > 0.0 ? vertexIntegrals.at(vertex) / volume : 0.0;
            }
            result.at(cell) = densitySum / static_cast<double>(mesh.numberOfCellVertices);
        }
        return result;
    }
} // namespace hase::kernels
