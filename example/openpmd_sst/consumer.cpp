#include <openPMD/openPMD.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace io = openPMD;

template<typename T>
std::vector<T> loadScalarMesh(io::Series& series, io::Iteration& iteration, std::string const& recordName)
{
    auto component = iteration.meshes[recordName][io::MeshRecordComponent::SCALAR];
    auto extent = component.getExtent();
    std::size_t elementCount
        = std::accumulate(extent.begin(), extent.end(), std::size_t{1}, std::multiplies<std::size_t>{});

    auto chunk = component.loadChunk<T>();
    series.flush();

    return std::vector<T>(chunk.get(), chunk.get() + elementCount);
}

template<typename T>
std::vector<T> loadMeshComponent(
    io::Series& series,
    io::Iteration& iteration,
    std::string const& recordName,
    std::string const& componentName)
{
    auto component = iteration.meshes[recordName][componentName];
    auto extent = component.getExtent();
    std::size_t elementCount
        = std::accumulate(extent.begin(), extent.end(), std::size_t{1}, std::multiplies<std::size_t>{});

    auto chunk = component.loadChunk<T>();
    series.flush();

    return std::vector<T>(chunk.get(), chunk.get() + elementCount);
}

int main(int argc, char** argv)
{
#if openPMD_HAVE_ADIOS2
    auto backends = io::getFileExtensions();
    if(std::find(backends.begin(), backends.end(), "sst") == backends.end())
    {
        std::cout << "SST engine not available in this openPMD-api/ADIOS2 build.\n";
        return 0;
    }

    std::string stream = "hase_input.sst";
    if(argc > 1)
    {
        stream = argv[1];
    }

    io::Series series(stream, io::Access::READ_LINEAR);

    for(auto& [index, iteration] : series.snapshots())
    {
        auto vertexX = loadMeshComponent<double>(series, iteration, "core_vertices", "x");
        auto vertexY = loadMeshComponent<double>(series, iteration, "core_vertices", "y");
        auto connectivity = loadScalarMesh<unsigned>(series, iteration, "core_connectivity");
        auto neighbors = loadScalarMesh<int>(series, iteration, "core_neighbors");
        auto cellCenterX = loadMeshComponent<double>(series, iteration, "core_cell_center", "x");
        auto cellCenterY = loadMeshComponent<double>(series, iteration, "core_cell_center", "y");
        auto betaVolume = loadScalarMesh<double>(series, iteration, "core_beta_volume");
        auto claddingCellType = loadScalarMesh<unsigned>(series, iteration, "core_cladding_cell_type");
        auto reflectivity = loadScalarMesh<float>(series, iteration, "core_reflectivity");

        std::cout << "Read openPMD/SST iteration " << index << "\n";
        std::cout << "  vertices           : " << vertexX.size() << " x, " << vertexY.size() << " y\n";
        std::cout << "  connectivity values: " << connectivity.size() << "\n";
        std::cout << "  neighbor values    : " << neighbors.size() << "\n";
        std::cout << "  cell centers       : " << cellCenterX.size() << " x, " << cellCenterY.size() << " y\n";
        std::cout << "  beta volume values : " << betaVolume.size() << "\n";
        std::cout << "  cladding values    : " << claddingCellType.size() << "\n";
        std::cout << "  reflectivity values: " << reflectivity.size() << "\n";

        iteration.close();
        break;
    }

    series.close();
    return 0;
#else
    (void) argc;
    (void) argv;
    std::cout << "This example requires openPMD-api built with ADIOS2 support.\n";
    return 0;
#endif
}
