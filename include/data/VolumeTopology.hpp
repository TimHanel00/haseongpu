#pragma once

#include <transport/TransportReader.hpp>

#include <cstdint>
#include <string>

namespace hase::data
{
    /**
     * @brief Host-side topology arrays read directly from the primitive graph.
     *
     * Preparation projects these arrays once into resident HybridBuffers.
     * They remain independent of material spectra so synchronized material
     * updates do not rebuild or upload topology.
     */
    class VolumeTopology
    {
    public:
        struct FieldName
        {
            static constexpr char const* points = "points";
            static constexpr char const* cellPointIndices = "cellPointIndices";
            static constexpr char const* cellTypes = "cellTypes";
            static constexpr char const* cellDomains = "cellDomains";
            static constexpr char const* facePointIndices = "facePointIndices";
            static constexpr char const* neighborCells = "neighborCells";
            static constexpr char const* neighborLocalFaces = "neighborLocalFaces";
            static constexpr char const* faceBoundaries = "faceBoundaries";
            static constexpr char const* faceCenters = "faceCenters";
            static constexpr char const* faceNormals = "faceNormals";
            static constexpr char const* faceAreas = "faceAreas";
            static constexpr char const* cellCenters = "cellCenters";
            static constexpr char const* cellVolumes = "cellVolumes";
            static constexpr char const* samplePoints = "samplePoints";
            static constexpr char const* metadata = "metadata";
        };

        transport::Array<double> points;
        transport::Array<std::uint32_t> cellPointIndices;
        transport::Array<std::uint32_t> cellTypes;
        transport::Array<std::int32_t> cellDomains;
        transport::Array<std::int32_t> facePointIndices;
        transport::Array<std::int32_t> neighborCells;
        transport::Array<std::int32_t> neighborLocalFaces;
        transport::Array<std::int32_t> faceBoundaries;
        transport::Array<double> faceCenters;
        transport::Array<double> faceNormals;
        transport::Array<double> faceAreas;
        transport::Array<double> cellCenters;
        transport::Array<double> cellVolumes;
        transport::Array<double> samplePoints;
        std::string metadata;

        /**
         * @brief Read one complete host-side topology node.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the topology node.
         * @return Topology arrays converted to transport SI units where applicable.
         */
        static VolumeTopology fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
