#include <data/VolumeTopology.hpp>

namespace hase::data
{
    VolumeTopology VolumeTopology::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        VolumeTopology result;
        reader.assign(result.points, prefix, FieldName::points);
        reader.assign(result.cellPointIndices, prefix, FieldName::cellPointIndices);
        reader.assign(result.cellTypes, prefix, FieldName::cellTypes);
        reader.assign(result.cellDomains, prefix, FieldName::cellDomains);
        reader.assign(result.facePointIndices, prefix, FieldName::facePointIndices);
        reader.assign(result.neighborCells, prefix, FieldName::neighborCells);
        reader.assign(result.neighborLocalFaces, prefix, FieldName::neighborLocalFaces);
        reader.assign(result.faceBoundaries, prefix, FieldName::faceBoundaries);
        reader.assign(result.faceCenters, prefix, FieldName::faceCenters);
        reader.assign(result.faceNormals, prefix, FieldName::faceNormals);
        reader.assign(result.faceAreas, prefix, FieldName::faceAreas);
        reader.assign(result.cellCenters, prefix, FieldName::cellCenters);
        reader.assign(result.cellVolumes, prefix, FieldName::cellVolumes);
        reader.assign(result.samplePoints, prefix, FieldName::samplePoints);
        reader.assign(result.metadata, prefix, FieldName::metadata);
        return result;
    }
} // namespace hase::data
