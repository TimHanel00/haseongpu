#include <openPMD/openPMD.hpp>
#include <openpmd/OpenPmdParser.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
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
  "backend": "adios2",
  "adios2": {
    "engine": {
      "type": "sst",
      "parameters": {
        "DataTransport": "WAN",
        "OpenTimeoutSecs": "600",
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

    std::vector<std::string> splitAxesString(std::string const& value)
    {
        std::vector<std::string> axes;
        std::stringstream stream(value);
        std::string axis;
        while(std::getline(stream, axis, ','))
        {
            if(!axis.empty())
            {
                axes.push_back(axis);
            }
        }
        return axes;
    }

    void validateAxesAttribute(
        std::string const& name,
        io::Attributable const& obj,
        std::vector<std::string> const& expected)
    {
        if(obj.containsAttribute("haseAxes"))
        {
            try
            {
                if(attribute<std::vector<std::string>>(obj, "haseAxes") == expected)
                {
                    return;
                }
            }
            catch(std::exception const&)
            {
            }
        }

        if(obj.containsAttribute("haseAxesString"))
        {
            // SST can carry Python string-list attributes as an empty scalar
            // string. Accept the scalar fallback while keeping haseAxes as the
            // canonical metadata for file-backed backends.
            if(splitAxesString(attribute<std::string>(obj, "haseAxesString")) == expected)
            {
                return;
            }
        }

        validationError(name, "attribute 'haseAxes' mismatch");
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
        validateAxesAttribute(name, record, axes);
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
        if(actual == expected)
        {
            return;
        }
        if(record.containsAttribute("haseAxisLabelsString"))
        {
            // SST can drop openPMD axisLabels on streamed mesh records. Accept
            // the scalar fallback while keeping axisLabels canonical elsewhere.
            if(splitAxesString(attribute<std::string>(record, "haseAxisLabelsString")) == expected)
            {
                return;
            }
        }
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
        auto const numberOfSamples = mesh.numberOfSamples;
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
        record.setAttribute("geometryParameters", "entity=sample_point");
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

    core::SimulationContext Parser::read()
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
            auto context = readIteration(series, iteration);
            series.close();
            return context;
        }

        throw std::runtime_error("No iteration was available in the openPMD input stream.");
    }

    bool Parser::hasStaticMeshUpdate(io::Iteration const& iteration) const
    {
        std::string const prefix = m_meshGroup + "_";
        bool const hasCanonical = iteration.meshes.contains(prefix + "points")
                                  || iteration.meshes.contains(prefix + "cells_connectivity")
                                  || iteration.meshes.contains(prefix + "cells_offsets")
                                  || iteration.meshes.contains(prefix + "cells_types");
        if(hasCanonical)
        {
            bool const completeCanonical = iteration.meshes.contains(prefix + "points")
                                           && iteration.meshes.contains(prefix + "cells_connectivity")
                                           && iteration.meshes.contains(prefix + "cells_offsets")
                                           && iteration.meshes.contains(prefix + "cells_types");
            if(!completeCanonical)
            {
                validationError("canonical topology", "partial static topology update");
            }
            return true;
        }

        bool const hasOldTopology
            = iteration.meshes.contains(prefix + "vertices") || iteration.meshes.contains(prefix + "connectivity");
        if(hasOldTopology)
        {
            validationError("explicit topology", "backend requires explicit 3D unstructured cell records");
        }
        return false;
    }

    core::Point pointAt(std::vector<double> const& points, unsigned numberOfPoints, unsigned pointIndex)
    {
        return core::Point{
            points.at(pointIndex),
            points.at(pointIndex + numberOfPoints),
            points.at(pointIndex + 2u * numberOfPoints)};
    }

    double tetraVolume(core::Point const a, core::Point const b, core::Point const c, core::Point const d)
    {
        return std::abs(core::dot(core::cross(b - a, c - a), d - a)) / 6.0;
    }

    std::vector<double> deriveCellCenters(
        std::vector<double> const& points,
        std::vector<unsigned> const& connectivity,
        unsigned numberOfPoints,
        unsigned numberOfCells)
    {
        std::vector<double> centers(3u * numberOfCells, 0.0);
        for(unsigned cell = 0u; cell < numberOfCells; ++cell)
        {
            core::Point center{0.0, 0.0, 0.0};
            for(unsigned localVertex = 0u; localVertex < core::tet4VertexCount; ++localVertex)
            {
                center = center + pointAt(points, numberOfPoints, connectivity.at(cell * core::tet4VertexCount + localVertex));
            }
            center = center * (1.0 / static_cast<double>(core::tet4VertexCount));
            centers.at(cell) = center.x;
            centers.at(cell + numberOfCells) = center.y;
            centers.at(cell + 2u * numberOfCells) = center.z;
        }
        return centers;
    }

    std::vector<float> deriveCellVolumes(
        std::vector<double> const& points,
        std::vector<unsigned> const& connectivity,
        unsigned numberOfPoints,
        unsigned numberOfCells)
    {
        std::vector<float> volumes(numberOfCells, 0.0f);
        for(unsigned cell = 0u; cell < numberOfCells; ++cell)
        {
            std::array<core::Point, core::tet4VertexCount> p{};
            for(unsigned localVertex = 0u; localVertex < core::tet4VertexCount; ++localVertex)
            {
                p.at(localVertex)
                    = pointAt(points, numberOfPoints, connectivity.at(cell * core::tet4VertexCount + localVertex));
            }
            volumes.at(cell) = static_cast<float>(tetraVolume(p[0], p[1], p[2], p[3]));
        }
        return volumes;
    }

    unsigned componentExtent(io::Iteration& iteration, std::string const& name, std::string const& componentName)
    {
        if(!iteration.meshes.contains(name) || !iteration.meshes[name].contains(componentName))
        {
            validationError(name + "/" + componentName, "missing required record component");
        }
        auto const extent = iteration.meshes[name][componentName].getExtent();
        if(extent.size() != 1u)
        {
            validationError(name + "/" + componentName, "expected one-dimensional component");
        }
        return static_cast<unsigned>(extent.at(0));
    }

    core::SimulationContext Parser::readIteration(io::Series& series, io::Iteration& iteration)
    {
        std::string const prefix = m_meshGroup + "_";

        auto const numberOfPoints = attribute<unsigned>(iteration, field::numberOfPoints);
        auto const numberOfCells = attribute<unsigned>(iteration, field::numberOfCells);
        if(!iteration.meshes.contains(prefix + "points") || !iteration.meshes.contains(prefix + "cell_faces")
           || !iteration.meshes.contains(prefix + "cell_neighbor_cells")
           || !iteration.meshes.contains(prefix + "cell_neighbor_local_faces")
           || !iteration.meshes.contains(prefix + "sample_points"))
        {
            validationError("explicit topology", "backend requires explicit 3D unstructured cell records");
        }

        std::vector<double> points = concatenate(
            concatenate(
                loadComponent<double>(series, iteration, prefix + "points", "x", io::Extent{numberOfPoints}),
                loadComponent<double>(series, iteration, prefix + "points", "y", io::Extent{numberOfPoints})),
            loadComponent<double>(series, iteration, prefix + "points", "z", io::Extent{numberOfPoints}));

        auto connectivity = loadScalar<unsigned>(
            series,
            iteration,
            prefix + "cells_connectivity",
            io::Extent{core::tet4VertexCount * numberOfCells},
            {"cell", "local_vertex"},
            {numberOfCells, core::tet4VertexCount},
            false,
            false);
        auto offsets = loadScalar<unsigned>(
            series,
            iteration,
            prefix + "cells_offsets",
            io::Extent{numberOfCells + 1u},
            {"cell_offset"},
            {numberOfCells + 1u},
            false,
            false);
        auto cellTypes = loadScalar<unsigned>(
            series,
            iteration,
            prefix + "cells_types",
            io::Extent{numberOfCells},
            {"cell"},
            {numberOfCells},
            false,
            false);
        for(unsigned cell = 0u; cell < numberOfCells; ++cell)
        {
            if(offsets.at(cell) != core::tet4VertexCount * cell || cellTypes.at(cell) != core::vtkTetraCellType)
            {
                validationError("explicit topology", "only contiguous VTK_TETRA Tet4 cells are supported");
            }
        }
        if(offsets.back() != core::tet4VertexCount * numberOfCells)
        {
            validationError("explicit topology", "cell offsets do not match Tet4 connectivity");
        }

        unsigned const numberOfSamples = componentExtent(iteration, prefix + "sample_points", "x");
        std::vector<double> samplePoints = concatenate(
            concatenate(
                loadComponent<double>(series, iteration, prefix + "sample_points", "x", io::Extent{numberOfSamples}),
                loadComponent<double>(series, iteration, prefix + "sample_points", "y", io::Extent{numberOfSamples})),
            loadComponent<double>(series, iteration, prefix + "sample_points", "z", io::Extent{numberOfSamples}));
        auto cellVolumes = deriveCellVolumes(points, connectivity, numberOfPoints, numberOfCells);
        auto cellCenters = deriveCellCenters(points, connectivity, numberOfPoints, numberOfCells);

        core::HostMesh mesh(
            std::move(connectivity),
            std::move(cellTypes),
            loadScalar<int>(
                series,
                iteration,
                prefix + "cell_faces",
                io::Extent{numberOfCells * core::tet4FaceCount * core::tet4FaceWidth},
                {"cell", "local_face", "local_vertex"},
                {numberOfCells, core::tet4FaceCount, core::tet4FaceWidth},
                false,
                false),
            loadScalar<int>(
                series,
                iteration,
                prefix + "cell_neighbor_cells",
                io::Extent{numberOfCells * core::tet4FaceCount},
                {"cell", "local_face"},
                {numberOfCells, core::tet4FaceCount},
                false,
                false),
            loadScalar<int>(
                series,
                iteration,
                prefix + "cell_neighbor_local_faces",
                io::Extent{numberOfCells * core::tet4FaceCount},
                {"cell", "local_face"},
                {numberOfCells, core::tet4FaceCount},
                false,
                false),
            loadScalar<int>(
                series,
                iteration,
                prefix + "cell_face_boundaries",
                io::Extent{numberOfCells * core::tet4FaceCount},
                {"cell", "local_face"},
                {numberOfCells, core::tet4FaceCount},
                false,
                false),
            std::move(cellVolumes),
            std::move(points),
            std::move(samplePoints),
            std::move(cellCenters),
            loadScalar<double>(
                series,
                iteration,
                prefix + "beta_volume",
                io::Extent{numberOfCells},
                {"cell"},
                {numberOfCells},
                true,
                true),
            loadScalar<double>(
                series,
                iteration,
                prefix + "point_beta",
                io::Extent{numberOfSamples},
                {"sample_point"},
                {numberOfSamples},
                true,
                true),
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
        return {std::move(experiment), std::move(compute), std::move(mesh), std::move(result)};
    }

    void Parser::updateDynamicIteration(
        io::Series& series,
        io::Iteration& iteration,
        core::SimulationContext& simulation)
    {
        std::string const prefix = m_meshGroup + "_";
        auto const numberOfCells = simulation.mesh.numberOfCells;
        auto const numberOfSamples = simulation.mesh.numberOfSamples;
        simulation.mesh.betaVolume = loadScalar<double>(
            series,
            iteration,
            prefix + "beta_volume",
            io::Extent{numberOfCells},
            {"cell"},
            {numberOfCells},
            true,
            true);
        simulation.mesh.betaCells = loadScalar<double>(
            series,
            iteration,
            prefix + "point_beta",
            io::Extent{numberOfSamples},
            {"sample_point"},
            {numberOfSamples},
            true,
            true);
        initializeResultForMesh(simulation.result, simulation.mesh);
        iteration.close();
    }

    void Parser::writeResult(core::Result const& result, core::HostMesh const& mesh)
    {
        if(!isHeadRank())
        {
            return;
        }

        auto const stream = m_outputPath.string();
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        io::Series series(stream, io::Access::CREATE_LINEAR, m_comm, seriesConfig(stream));
#else
        io::Series series(stream, io::Access::CREATE_LINEAR, seriesConfig(stream));
#endif
        series.setAttribute("haseSchemaVersion", std::string{HASE_SCHEMA_VERSION});
        writeResultIteration(series, 0u, result, mesh);
        series.close();
    }

    void Parser::writeResultIteration(
        io::Series& series,
        std::uint64_t iterationIndex,
        core::Result const& result,
        core::HostMesh const& mesh)
    {
        auto const extent = io::Extent{mesh.numberOfSamples};
        auto iterations = series.writeIterations();
        auto iteration = iterations[iterationIndex];
        iteration.setTime(0.0);
        iteration.setDt(1.0);
        iteration.setTimeUnitSI(1.0);
        iteration.setAttribute(field::numberOfPoints, mesh.numberOfPoints);
        iteration.setAttribute(field::numberOfCells, mesh.numberOfCells);

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
            {"sample_point"},
            "cm^-2 s^-1",
            1.0e4,
            {-2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0});
        writeScalar(iteration, prefix + "mse", mse, extent, {"sample_point"});
        writeScalar(iteration, prefix + "total_rays", totalRays, extent, {"sample_point"}, "count");
        writeScalar(
            iteration,
            prefix + "dndt_ase",
            dndtAse,
            extent,
            {"sample_point"},
            "s^-1",
            1.0,
            {0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0});

        iteration.close();
    }

    void Parser::processAll(std::function<void(core::SimulationContext&)> process)
    {
        auto const inputStream = m_inputPath.string();
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        io::Series inputSeries(inputStream, io::Access::READ_LINEAR, m_comm, seriesConfig(inputStream));
#else
        io::Series inputSeries(inputStream, io::Access::READ_LINEAR, seriesConfig(inputStream));
#endif

        std::unique_ptr<io::Series> outputSeries;
        if(isHeadRank())
        {
            auto const outputStream = m_outputPath.string();
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
            outputSeries = std::make_unique<io::Series>(
                outputStream,
                io::Access::CREATE_LINEAR,
                m_comm,
                seriesConfig(outputStream));
#else
            outputSeries
                = std::make_unique<io::Series>(outputStream, io::Access::CREATE_LINEAR, seriesConfig(outputStream));
#endif
            outputSeries->setAttribute("haseSchemaVersion", std::string{HASE_SCHEMA_VERSION});
        }

        std::optional<core::SimulationContext> simulation;
        for(auto iteration : inputSeries.readIterations())
        {
            auto const iterationIndex = iteration.iterationIndex;
            if(hasStaticMeshUpdate(iteration))
            {
                simulation = readIteration(inputSeries, iteration);
            }
            else
            {
                if(!simulation)
                {
                    validationError("dynamic iteration", "arrived before any static mesh update");
                }
                updateDynamicIteration(inputSeries, iteration, *simulation);
            }

            process(*simulation);
            if(outputSeries)
            {
                writeResultIteration(*outputSeries, iterationIndex, simulation->result, simulation->mesh);
            }
        }

        inputSeries.close();
        if(outputSeries)
        {
            outputSeries->close();
        }
    }

} // namespace hase::openpmd
