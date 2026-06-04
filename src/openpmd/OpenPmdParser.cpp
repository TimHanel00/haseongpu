#include <openPMD/openPMD.hpp>
#include <openpmd/OpenPmdParser.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
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
        constexpr char const* rngSeed = "rng_seed";
        constexpr char const* writeVtk = "write_vtk";
        constexpr char const* devices = "devices";
    } // namespace field

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
    constexpr char const* HASE_SCHEMA_VERSION = "0.1";

    bool hasSuffix(std::string_view value, std::string_view suffix)
    {
        return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
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

    [[noreturn]] void validationError(std::string const& field, std::string const& message)
    {
        throw std::runtime_error("openPMD validation error for '" + field + "': " + message);
    }

    std::string entityFromAxes(std::vector<std::string> const& axes);

    template<typename T>
    std::string vectorString(std::vector<T> const& values)
    {
        std::ostringstream out;
        out << "[";
        for(std::size_t i = 0; i < values.size(); ++i)
        {
            if(i != 0)
            {
                out << ", ";
            }
            out << values[i];
        }
        out << "]";
        return out.str();
    }

    std::vector<unsigned long long> extentVector(io::Extent const& extent)
    {
        return std::vector<unsigned long long>(extent.begin(), extent.end());
    }

    void validateExtent(std::string const& name, io::Extent const& extent, io::Extent const& expected)
    {
        if(extent != expected)
        {
            auto const actualVector = extentVector(extent);
            auto const expectedVector = extentVector(expected);
            validationError(
                name,
                "extent mismatch (expected " + vectorString(expectedVector) + ", got " + vectorString(actualVector)
                    + ")");
        }
    }

    template<typename T>
    void validateAttribute(
        std::string const& name,
        io::Attributable const& obj,
        std::string const& attributeName,
        T const& expected)
    {
        if(!obj.containsAttribute(attributeName))
        {
            validationError(name, "missing required attribute '" + attributeName + "'");
        }
        auto const actual = attribute<T>(obj, attributeName);
        if(actual != expected)
        {
            validationError(name, "attribute '" + attributeName + "' mismatch");
        }
    }

    void validateHaseMetadata(
        std::string const& name,
        io::Mesh const& record,
        std::vector<std::string> const& axes,
        std::vector<unsigned long long> const& primitiveShape,
        bool dynamic,
        bool backendRequired,
        std::string const& unit)
    {
        validateAttribute(name, record, "haseSchemaVersion", std::string{HASE_SCHEMA_VERSION});
        validateAttribute(name, record, "haseEntity", entityFromAxes(axes));
        validateAttribute(name, record, "haseAxes", axes);
        validateAttribute(name, record, "haseLayoutOrder", std::string{"backendFlat"});
        validateAttribute(name, record, "hasePrimitiveShape", primitiveShape);
        validateAttribute(name, record, "haseStatic", !dynamic);
        validateAttribute(name, record, "haseDynamic", dynamic);
        validateAttribute(name, record, "haseBackendRequired", backendRequired);
        validateAttribute(name, record, "haseUnit", unit);
    }

    void validateAxisLabels(std::string const& name, io::Mesh const& record, std::vector<std::string> const& expected)
    {
        auto const actual = record.axisLabels();
        if(actual != expected)
        {
            validationError(
                name,
                "axis labels mismatch (expected " + vectorString(expected) + ", got " + vectorString(actual) + ")");
        }
    }

    template<typename T>
    std::vector<T> loadScalar(
        io::Series& series,
        io::Iteration& iteration,
        std::string const& name,
        io::Extent const& expectedExtent,
        std::vector<std::string> const& axes,
        std::vector<unsigned long long> const& primitiveShape,
        bool dynamic,
        bool backendRequired,
        std::string const& unit = "1")
    {
        if(!iteration.meshes.contains(name))
        {
            validationError(name, "missing required mesh record");
        }
        auto record = iteration.meshes[name];
        validateHaseMetadata(name, record, axes, primitiveShape, dynamic, backendRequired, unit);
        validateAxisLabels(name, record, {"flatIndex"});
        if(!record.contains(io::MeshRecordComponent::SCALAR))
        {
            validationError(name, "missing required scalar component");
        }
        auto component = record[io::MeshRecordComponent::SCALAR];
        auto const expectedDatatype = io::determineDatatype<T>();
        auto const actualDatatype = component.getDatatype();
        if(actualDatatype != expectedDatatype)
        {
            validationError(
                name,
                "dtype mismatch (expected " + io::datatypeToString(expectedDatatype) + ", got "
                    + io::datatypeToString(actualDatatype) + ")");
        }
        auto const extent = component.getExtent();
        validateExtent(name, extent, expectedExtent);
        auto chunk = component.loadChunk<T>();
        series.flush();
        return std::vector<T>(chunk.get(), chunk.get() + elementCount(extent));
    }

    template<typename T>
    std::vector<T> loadComponent(
        io::Series& series,
        io::Iteration& iteration,
        std::string const& name,
        std::string const& componentName,
        io::Extent const& expectedExtent)
    {
        if(!iteration.meshes.contains(name))
        {
            validationError(name, "missing required mesh record");
        }
        auto record = iteration.meshes[name];
        if(!record.contains(componentName))
        {
            validationError(name + "/" + componentName, "missing required record component");
        }
        auto component = record[componentName];
        auto const expectedDatatype = io::determineDatatype<T>();
        auto const actualDatatype = component.getDatatype();
        if(actualDatatype != expectedDatatype)
        {
            validationError(
                name + "/" + componentName,
                "dtype mismatch (expected " + io::datatypeToString(expectedDatatype) + ", got "
                    + io::datatypeToString(actualDatatype) + ")");
        }
        auto const extent = component.getExtent();
        validateExtent(name + "/" + componentName, extent, expectedExtent);
        auto chunk = component.loadChunk<T>();
        series.flush();
        return std::vector<T>(chunk.get(), chunk.get() + elementCount(extent));
    }

    void validateComputeSettings(io::Iteration const& iteration)
    {
        if(iteration.containsAttribute(field::writeVtk) && attribute<bool>(iteration, field::writeVtk))
        {
            validationError(
                "compute/" + std::string{field::writeVtk},
                "unsupported compute setting; openPMD parser rejects VTK output requests");
        }
        if(iteration.containsAttribute(field::devices))
        {
            validationError(
                "compute/" + std::string{field::devices},
                "unsupported compute setting; explicit device lists are not preserved by this transport");
        }
    }

    std::string entityFromAxes(std::vector<std::string> const& axes)
    {
        std::string entity;
        for(auto const& axis : axes)
        {
            if(!entity.empty())
            {
                entity += "_";
            }
            entity += axis;
        }
        return entity;
    }

    std::vector<unsigned long long> shapeFromExtent(io::Extent const& extent)
    {
        return std::vector<unsigned long long>(extent.begin(), extent.end());
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
        std::vector<std::string> const& axisLabels,
        std::string const& unit = "1",
        double unitSI = 1.0,
        std::array<double, 7> unitDimension = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
    {
        auto record = iteration.meshes[name];
        record.setAttribute("geometry", "other");
        record.setAttribute("geometryParameters", "entity=point_level");
        record.setAttribute("dataOrder", "C");
        record.setAxisLabels(axisLabels);
        record.setAttribute("haseSchemaVersion", std::string{HASE_SCHEMA_VERSION});
        record.setAttribute("haseEntity", entityFromAxes(axisLabels));
        record.setAttribute("haseAxes", axisLabels);
        record.setAttribute("haseLayoutOrder", std::string{"recordC"});
        record.setAttribute("hasePrimitiveShape", shapeFromExtent(extent));
        record.setAttribute("haseStatic", false);
        record.setAttribute("haseDynamic", true);
        record.setAttribute("haseBackendRequired", false);
        record.setAttribute("haseUnit", unit);
        record.setGridSpacing(std::vector<double>(extent.size(), 1.0));
        record.setGridGlobalOffset(std::vector<double>(extent.size(), 0.0));
        record.setGridUnitSI(1.0);
        record.setUnitDimension(unitDimension);

        auto& component = record[io::MeshRecordComponent::SCALAR];
        component.setUnitSI(unitSI);
        component.setPosition(std::vector<double>(extent.size(), 0.0));
        component.resetDataset({io::determineDatatype<T>(), extent});
        component.storeChunk(values, io::Offset(extent.size(), 0u), extent);
    }
} // namespace

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

            auto vertices = iteration.meshes[prefix + "vertices"];
            validateHaseMetadata(
                prefix + "vertices",
                vertices,
                {"coordinate", "point"},
                {2u, numberOfPoints},
                false,
                true,
                "m");
            validateAxisLabels(prefix + "vertices", vertices, {"point"});
            auto points = concatenate(
                loadComponent<double>(series, iteration, prefix + "vertices", "x", io::Extent{numberOfPoints}),
                loadComponent<double>(series, iteration, prefix + "vertices", "y", io::Extent{numberOfPoints}));

            auto trianglePointIndices = loadScalar<unsigned>(
                series,
                iteration,
                prefix + "connectivity",
                io::Extent{3u * numberOfCells},
                {"cell", "local_vertex"},
                {numberOfCells, 3u},
                false,
                true);
            auto triangleNeighbors = loadScalar<int>(
                series,
                iteration,
                prefix + "neighbors",
                io::Extent{3u * numberOfCells},
                {"cell", "local_side"},
                {numberOfCells, 3u},
                false,
                true);
            auto forbiddenEdge = loadScalar<int>(
                series,
                iteration,
                prefix + "forbidden_edges",
                io::Extent{3u * numberOfCells},
                {"cell", "local_side"},
                {numberOfCells, 3u},
                false,
                true);
            auto triangleNormalPoint = loadScalar<unsigned>(
                series,
                iteration,
                prefix + "normal_points",
                io::Extent{3u * numberOfCells},
                {"cell", "local_side"},
                {numberOfCells, 3u},
                false,
                true);
            auto triangleNormalsX = loadScalar<double>(
                series,
                iteration,
                prefix + "cell_normal_x",
                io::Extent{3u * numberOfCells},
                {"cell", "local_side"},
                {numberOfCells, 3u},
                false,
                true);
            auto triangleNormalsY = loadScalar<double>(
                series,
                iteration,
                prefix + "cell_normal_y",
                io::Extent{3u * numberOfCells},
                {"cell", "local_side"},
                {numberOfCells, 3u},
                false,
                true);
            auto betaVolume = loadScalar<double>(
                series,
                iteration,
                prefix + "beta_volume",
                io::Extent{numberOfCells * (numberOfLevels - 1u)},
                {"cell", "layer"},
                {numberOfCells, numberOfLevels - 1u},
                true,
                true);
            auto betaCells = loadScalar<double>(
                series,
                iteration,
                prefix + "point_beta",
                io::Extent{numberOfPoints * numberOfLevels},
                {"point", "level"},
                {numberOfPoints, numberOfLevels},
                true,
                true);

            auto cellCenter = iteration.meshes[prefix + "cell_center"];
            validateHaseMetadata(prefix + "cell_center", cellCenter, {"cell"}, {numberOfCells}, false, true, "m");
            validateAxisLabels(prefix + "cell_center", cellCenter, {"cell"});

            core::HostMesh mesh(
                std::move(trianglePointIndices),
                numberOfCells,
                numberOfLevels,
                numberOfPoints,
                attribute<float>(iteration, field::thickness),
                std::move(points),
                loadComponent<double>(series, iteration, prefix + "cell_center", "x", io::Extent{numberOfCells}),
                loadComponent<double>(series, iteration, prefix + "cell_center", "y", io::Extent{numberOfCells}),
                std::move(triangleNormalPoint),
                std::move(triangleNormalsX),
                std::move(triangleNormalsY),
                std::move(forbiddenEdge),
                std::move(triangleNeighbors),
                loadScalar<float>(
                    series,
                    iteration,
                    prefix + "surface",
                    io::Extent{numberOfCells},
                    {"cell"},
                    {numberOfCells},
                    false,
                    true,
                    "m^2"),
                std::move(betaVolume),
                std::move(betaCells),
                loadScalar<unsigned>(
                    series,
                    iteration,
                    prefix + "cladding_cell_type",
                    io::Extent{numberOfCells},
                    {"cell"},
                    {numberOfCells},
                    false,
                    true),
                loadScalar<float>(
                    series,
                    iteration,
                    prefix + "refractive_index",
                    io::Extent{4u},
                    {"interface"},
                    {4u},
                    false,
                    true),
                loadScalar<float>(
                    series,
                    iteration,
                    prefix + "reflectivity",
                    io::Extent{2u * numberOfCells},
                    {"cell", "interface"},
                    {numberOfCells, 2u},
                    false,
                    true),
                attribute<float>(iteration, field::nTot),
                attribute<float>(iteration, field::crystalTFluo),
                attribute<unsigned>(iteration, field::claddingNumber),
                attribute<double>(iteration, field::claddingAbsorption));

            core::ExperimentParameters experiment(
                attribute<unsigned>(iteration, field::minRaysPerSample),
                attribute<unsigned>(iteration, field::maxRaysPerSample),
                loadScalar<double>(
                    series,
                    iteration,
                    prefix + "lambda_absorption",
                    io::Extent{attribute<unsigned>(iteration, field::spectralResolution)},
                    {"wavelength"},
                    {attribute<unsigned>(iteration, field::spectralResolution)},
                    false,
                    false,
                    "m"),
                loadScalar<double>(
                    series,
                    iteration,
                    prefix + "lambda_emission",
                    io::Extent{attribute<unsigned>(iteration, field::spectralResolution)},
                    {"wavelength"},
                    {attribute<unsigned>(iteration, field::spectralResolution)},
                    false,
                    false,
                    "m"),
                loadScalar<double>(
                    series,
                    iteration,
                    prefix + "sigma_absorption",
                    io::Extent{attribute<unsigned>(iteration, field::spectralResolution)},
                    {"wavelength"},
                    {attribute<unsigned>(iteration, field::spectralResolution)},
                    false,
                    false,
                    "cm^2"),
                loadScalar<double>(
                    series,
                    iteration,
                    prefix + "sigma_emission",
                    io::Extent{attribute<unsigned>(iteration, field::spectralResolution)},
                    {"wavelength"},
                    {attribute<unsigned>(iteration, field::spectralResolution)},
                    false,
                    false,
                    "cm^2"),
                0.0,
                0.0,
                attribute<double>(iteration, field::mseThreshold),
                attribute<bool>(iteration, field::useReflections),
                attribute<unsigned>(iteration, field::spectralResolution),
                attributeOr<bool>(iteration, field::monochromatic, false));

            experiment.maxSigmaA = attributeOr<double>(iteration, field::maxSigmaAbsorption, experiment.maxSigmaA);
            experiment.maxSigmaE = attributeOr<double>(iteration, field::maxSigmaEmission, experiment.maxSigmaE);

            validateComputeSettings(iteration);

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
                attributeOr<unsigned>(iteration, field::maxSampleRange, numberOfSamples - 1u),
                attributeOr<unsigned>(iteration, field::rngSeed, core::ComputeParameters::unspecifiedRngSeed));

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
        series.setAttribute("haseSchemaVersion", std::string{HASE_SCHEMA_VERSION});

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
        writeScalar(
            iteration,
            prefix + "phi_ase",
            phiAse,
            extent,
            {"point", "level"},
            "cm^-2 s^-1",
            1.0e4,
            {-2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0});
        writeScalar(iteration, prefix + "mse", mse, extent, {"point", "level"});
        writeScalar(iteration, prefix + "total_rays", totalRays, extent, {"point", "level"}, "count");
        writeScalar(
            iteration,
            prefix + "dndt_ase",
            dndtAse,
            extent,
            {"point", "level"},
            "s^-1",
            1.0,
            {0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0});

        iteration.close();
        series.close();
    }

} // namespace hase::openpmd
