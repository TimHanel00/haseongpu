#include <openpmd/OpenPmdParser.hpp>

#include <openPMD/openPMD.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace io = openPMD;

namespace
{
namespace field
{
    constexpr char const* numberOfPoints = "number_of_points";
    constexpr char const* numberOfCells = "number_of_cells";
    constexpr char const* numberOfLevels = "number_of_levels";
    constexpr char const* thickness = "thickness";
    constexpr char const* nTot = "n_tot";
    constexpr char const* crystalTFluo = "crystal_t_fluo";
    constexpr char const* claddingNumber = "cladding_number";
    constexpr char const* claddingAbsorption = "cladding_absorption";
    constexpr char const* minRaysPerSample = "min_rays_per_sample";
    constexpr char const* maxRaysPerSample = "max_rays_per_sample";
    constexpr char const* mseThreshold = "mse_threshold";
    constexpr char const* useReflections = "use_reflections";
    constexpr char const* spectralResolution = "spectral_resolution";
    constexpr char const* monochromatic = "monochromatic";
    constexpr char const* maxSigmaAbsorption = "max_sigma_absorption";
    constexpr char const* maxSigmaEmission = "max_sigma_emission";
    constexpr char const* repetitions = "repetitions";
    constexpr char const* adaptiveSteps = "adaptive_steps";
    constexpr char const* maxGpus = "max_gpus";
    constexpr char const* backend = "backend";
    constexpr char const* parallelMode = "parallel_mode";
    constexpr char const* minSampleRange = "min_sample_range";
    constexpr char const* maxSampleRange = "max_sample_range";
}

constexpr char const* OPENPMD_SST_CONFIG = R"(
{
  "adios2": {
    "engine": {
      "parameters": {
        "DataTransport": "WAN",
        "QueueFullPolicy": "Discard"
      }
    }
  }
})";

constexpr char const* OPENPMD_DEFAULT_CONFIG = "{}";

bool hasSuffix(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size()
        && value.substr(value.size() - suffix.size()) == suffix;
}

char const* seriesConfig(std::string const& stream)
{
    return hasSuffix(stream, ".sst") ? OPENPMD_SST_CONFIG : OPENPMD_DEFAULT_CONFIG;
}

template<typename T>
T attribute(io::Attributable const& obj, std::string const& name)
{
    return obj.getAttribute(name).get<T>();
}

template<typename T>
T attributeOr(io::Attributable const& obj, std::string const& name, T fallback)
{
    if(obj.containsAttribute(name))
    {
        return attribute<T>(obj, name);
    }
    return fallback;
}

std::size_t elementCount(io::Extent const& extent)
{
    return std::accumulate(extent.begin(), extent.end(), std::size_t{1}, std::multiplies<std::size_t>{});
}

template<typename T>
std::vector<T> loadScalar(io::Series& series, io::Iteration& iteration, std::string const& name)
{
    auto component = iteration.meshes[name][io::MeshRecordComponent::SCALAR];
    auto const extent = component.getExtent();
    auto chunk = component.loadChunk<T>();
    series.flush();
    return std::vector<T>(chunk.get(), chunk.get() + elementCount(extent));
}

template<typename T>
std::vector<T> loadComponent(
    io::Series& series,
    io::Iteration& iteration,
    std::string const& name,
    std::string const& componentName)
{
    auto component = iteration.meshes[name][componentName];
    auto const extent = component.getExtent();
    auto chunk = component.loadChunk<T>();
    series.flush();
    return std::vector<T>(chunk.get(), chunk.get() + elementCount(extent));
}

template<typename T>
std::vector<T> concatenate(std::vector<T> first, std::vector<T> const& second)
{
    first.insert(first.end(), second.begin(), second.end());
    return first;
}

void initializeResultForMesh(hase::core::Result& result, hase::core::HostMesh const& mesh)
{
    auto const numberOfSamples = mesh.numberOfPoints * mesh.numberOfLevels;
    result = hase::core::Result(
        std::vector<float>(numberOfSamples, 0.0f),
        std::vector<double>(numberOfSamples, 100000.0),
        std::vector<unsigned>(numberOfSamples, 0u),
        std::vector<double>(numberOfSamples, 0.0));
}

template<typename T>
void writeScalar(
    io::Iteration& iteration,
    std::string const& name,
    std::vector<T>& values,
    io::Extent const& extent,
    std::vector<std::string> const& axisLabels)
{
    auto record = iteration.meshes[name];
    record.setAttribute("geometry", "other");
    record.setAttribute("geometryParameters", "entity=point_level");
    record.setAttribute("dataOrder", "C");
    record.setAxisLabels(axisLabels);
    record.setGridSpacing(std::vector<double>(extent.size(), 1.0));
    record.setGridGlobalOffset(std::vector<double>(extent.size(), 0.0));
    record.setGridUnitSI(1.0);
    record.setUnitDimension(std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    auto& component = record[io::MeshRecordComponent::SCALAR];
    component.setUnitSI(1.0);
    component.setPosition(std::vector<double>(extent.size(), 0.0));
    component.resetDataset({io::determineDatatype<T>(), extent});
    component.storeChunk(values, io::Offset(extent.size(), 0u), extent);
}
}

namespace hase::openpmd
{

Parser::Parser(std::filesystem::path inputPath, std::filesystem::path outputPath)
    : m_inputPath(std::move(inputPath))
    , m_outputPath(std::move(outputPath))
{
}

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
Parser::Parser(std::filesystem::path inputPath, std::filesystem::path outputPath, MPI_Comm comm)
    : m_inputPath(std::move(inputPath))
    , m_outputPath(std::move(outputPath))
    , m_comm(comm)
{
}
#endif

bool Parser::isHeadRank() const
{
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
    int rank = 0;
    MPI_Comm_rank(m_comm, &rank);
    return rank == 0;
#else
    return true;
#endif
}

hase::openpmd::SimulationContext Parser::read()
{
    auto const stream = m_inputPath.string();
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
    io::Series series(stream, io::Access::READ_LINEAR, m_comm, seriesConfig(stream));
#else
    io::Series series(stream, io::Access::READ_LINEAR, seriesConfig(stream));
#endif

    for(auto& [index, iteration] : series.snapshots())
    {
        (void) index;
        std::string const prefix = m_meshGroup + "_";

        auto const numberOfPoints = attribute<unsigned>(iteration, field::numberOfPoints);
        auto const numberOfCells = attribute<unsigned>(iteration, field::numberOfCells);
        auto const numberOfLevels = attribute<unsigned>(iteration, field::numberOfLevels);

        auto points = concatenate(
            loadComponent<double>(series, iteration, prefix + "vertices", "x"),
            loadComponent<double>(series, iteration, prefix + "vertices", "y"));

        auto trianglePointIndices = loadScalar<unsigned>(series, iteration, prefix + "connectivity");
        auto triangleNeighbors = loadScalar<int>(series, iteration, prefix + "neighbors");
        auto forbiddenEdge = loadScalar<int>(series, iteration, prefix + "forbidden_edges");
        auto triangleNormalPoint = loadScalar<unsigned>(series, iteration, prefix + "normal_points");
        auto triangleNormalsX = loadScalar<double>(series, iteration, prefix + "cell_normal_x");
        auto triangleNormalsY = loadScalar<double>(series, iteration, prefix + "cell_normal_y");
        auto betaVolume = loadScalar<double>(series, iteration, prefix + "beta_volume");
        auto betaCells = loadScalar<double>(series, iteration, prefix + "point_beta");

        core::HostMesh mesh(
            std::move(trianglePointIndices),
            numberOfCells,
            numberOfLevels,
            numberOfPoints,
            attribute<float>(iteration, field::thickness),
            std::move(points),
            loadComponent<double>(series, iteration, prefix + "cell_center", "x"),
            loadComponent<double>(series, iteration, prefix + "cell_center", "y"),
            std::move(triangleNormalPoint),
            std::move(triangleNormalsX),
            std::move(triangleNormalsY),
            std::move(forbiddenEdge),
            std::move(triangleNeighbors),
            loadScalar<float>(series, iteration, prefix + "surface"),
            std::move(betaVolume),
            std::move(betaCells),
            loadScalar<unsigned>(series, iteration, prefix + "cladding_cell_type"),
            loadScalar<float>(series, iteration, prefix + "refractive_index"),
            loadScalar<float>(series, iteration, prefix + "reflectivity"),
            attribute<float>(iteration, field::nTot),
            attribute<float>(iteration, field::crystalTFluo),
            attribute<unsigned>(iteration, field::claddingNumber),
            attribute<double>(iteration, field::claddingAbsorption));

        core::ExperimentParameters experiment(
            attribute<unsigned>(iteration, field::minRaysPerSample),
            attribute<unsigned>(iteration, field::maxRaysPerSample),
            loadScalar<double>(series, iteration, prefix + "lambda_absorption"),
            loadScalar<double>(series, iteration, prefix + "lambda_emission"),
            loadScalar<double>(series, iteration, prefix + "sigma_absorption"),
            loadScalar<double>(series, iteration, prefix + "sigma_emission"),
            0.0,
            0.0,
            attribute<double>(iteration, field::mseThreshold),
            attribute<bool>(iteration, field::useReflections),
            attribute<unsigned>(iteration, field::spectralResolution),
            attributeOr<bool>(iteration, field::monochromatic, false));

        experiment.maxSigmaA = attributeOr<double>(iteration, field::maxSigmaAbsorption, experiment.maxSigmaA);
        experiment.maxSigmaE = attributeOr<double>(iteration, field::maxSigmaEmission, experiment.maxSigmaE);

        auto const numberOfSamples = numberOfPoints * numberOfLevels;
        core::ComputeParameters compute(
            attribute<unsigned>(iteration, field::repetitions),
            attribute<unsigned>(iteration, field::adaptiveSteps),
            attribute<unsigned>(iteration, field::maxGpus),
            0u,
            attribute<std::string>(iteration, field::backend),
            attributeOr<std::string>(iteration, field::parallelMode, core::ParallelMode::SINGLE),
            false,
            std::vector<unsigned>{},
            attributeOr<unsigned>(iteration, field::minSampleRange, 0u),
            attributeOr<unsigned>(iteration, field::maxSampleRange, numberOfSamples - 1u));

        mesh.calcTotalReflectionAngles();

        core::Result result;
        initializeResultForMesh(result, mesh);
        iteration.close();
        series.close();
        return {std::move(experiment), std::move(compute), std::move(mesh), std::move(result)};
    }

    throw std::runtime_error("No iteration was available in the openPMD input stream.");
}

void Parser::writeResult(core::Result const& result, core::HostMesh const& mesh)
{
    if(!isHeadRank())
    {
        return;
    }

    auto const extent = io::Extent{mesh.numberOfPoints, mesh.numberOfLevels};
    auto const stream = m_outputPath.string();

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
    io::Series series(stream, io::Access::CREATE_LINEAR, m_comm, seriesConfig(stream));
#else
    io::Series series(stream, io::Access::CREATE_LINEAR, seriesConfig(stream));
#endif

    auto iteration = series.snapshots()[0];
    iteration.setTime(0.0);
    iteration.setDt(1.0);
    iteration.setTimeUnitSI(1.0);
    iteration.setAttribute(field::numberOfPoints, mesh.numberOfPoints);
    iteration.setAttribute(field::numberOfLevels, mesh.numberOfLevels);

    std::string const prefix = m_meshGroup + "_result_";
    auto phiAse = result.phiAse;
    auto mse = result.mse;
    auto totalRays = result.totalRays;
    auto dndtAse = result.dndtAse;
    writeScalar(iteration, prefix + "phi_ase", phiAse, extent, {"point", "level"});
    writeScalar(iteration, prefix + "mse", mse, extent, {"point", "level"});
    writeScalar(iteration, prefix + "total_rays", totalRays, extent, {"point", "level"});
    writeScalar(iteration, prefix + "dndt_ase", dndtAse, extent, {"point", "level"});

    iteration.close();
    series.close();
}

} // namespace hase::openpmd
