#include <catch2/catch_test_macros.hpp>
#include <openPMD/openPMD.hpp>
#include <openpmd/OpenPmdParser.hpp>

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace io = openPMD;

namespace
{
#ifndef HASE_OPENPMD_FILE_EXTENSION
#    define HASE_OPENPMD_FILE_EXTENSION "bp"
#endif

    constexpr char const* HASE_SCHEMA_VERSION = "0.1";

    std::filesystem::path testPath(std::string const& name)
    {
        auto path
            = std::filesystem::temp_directory_path() / ("hase_openpmd_" + name + "." + HASE_OPENPMD_FILE_EXTENSION);
        std::filesystem::remove_all(path);
        return path;
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

    void setMetadata(
        io::Mesh& record,
        std::vector<std::string> const& axes,
        std::vector<unsigned long long> const& primitiveShape,
        bool dynamic = false,
        bool backendRequired = true,
        std::string const& unit = "1",
        std::vector<std::string> const& axisLabels = {"flatIndex"})
    {
        record.setAttribute("geometry", "other");
        record.setAttribute("geometryParameters", "topology=unstructured_triangular_prism");
        record.setAttribute("dataOrder", "C");
        record.setAxisLabels(axisLabels);
        record.setAttribute("haseSchemaVersion", std::string{HASE_SCHEMA_VERSION});
        record.setAttribute("haseEntity", entityFromAxes(axes));
        record.setAttribute("haseAxes", axes);
        record.setAttribute("haseLayoutOrder", std::string{"backendFlat"});
        record.setAttribute("hasePrimitiveShape", primitiveShape);
        record.setAttribute("haseStatic", !dynamic);
        record.setAttribute("haseDynamic", dynamic);
        record.setAttribute("haseBackendRequired", backendRequired);
        record.setAttribute("haseUnit", unit);
        record.setGridSpacing(std::vector<double>(axisLabels.size(), 1.0));
        record.setGridGlobalOffset(std::vector<double>(axisLabels.size(), 0.0));
        record.setGridUnitSI(1.0);
        record.setUnitDimension(std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    }

    template<typename T>
    void writeScalar(
        io::Series& series,
        io::Iteration& iteration,
        std::string const& name,
        std::vector<T> values,
        std::vector<std::string> const& axes,
        std::vector<unsigned long long> const& primitiveShape,
        bool dynamic = false,
        bool backendRequired = true,
        std::string const& unit = "1")
    {
        auto record = iteration.meshes[name];
        setMetadata(record, axes, primitiveShape, dynamic, backendRequired, unit);
        auto& component = record[io::MeshRecordComponent::SCALAR];
        io::Extent extent{values.size()};
        component.setUnitSI(1.0);
        component.setPosition(std::vector<double>{0.0});
        component.resetDataset({io::determineDatatype<T>(), extent});
        component.storeChunk(values, io::Offset{0u}, extent);
        series.flush();
    }

    template<typename T>
    void writeComponent(
        io::Series& series,
        io::Iteration& iteration,
        std::string const& recordName,
        std::string const& componentName,
        std::vector<T> values,
        std::vector<std::string> const& axisLabels)
    {
        auto record = iteration.meshes[recordName];
        record.setAttribute("geometry", "other");
        record.setAttribute("dataOrder", "C");
        record.setAxisLabels(axisLabels);
        record.setGridSpacing(std::vector<double>(axisLabels.size(), 1.0));
        record.setGridGlobalOffset(std::vector<double>(axisLabels.size(), 0.0));
        record.setGridUnitSI(1.0);
        auto& component = record[componentName];
        io::Extent extent{values.size()};
        component.setUnitSI(1.0);
        component.setPosition(std::vector<double>{0.0});
        component.resetDataset({io::determineDatatype<T>(), extent});
        component.storeChunk(values, io::Offset{0u}, extent);
        series.flush();
    }

    std::filesystem::path writeInput(
        std::string const& name,
        std::function<void(io::Series&, io::Iteration&)> mutate = {},
        bool betaVolumeAsFloat = false,
        bool pointBetaTransposed = false)
    {
        auto path = testPath(name);
        io::Series series(path.string(), io::Access::CREATE_LINEAR, "{}");
        series.setAttribute("haseSchemaVersion", std::string{HASE_SCHEMA_VERSION});
        auto iteration = series.snapshots()[0];
        iteration.setTime(0.0);
        iteration.setDt(1.0);
        iteration.setTimeUnitSI(1.0);

        iteration.setAttribute("number_of_points", 3u);
        iteration.setAttribute("number_of_cells", 1u);
        iteration.setAttribute("number_of_levels", 2u);
        iteration.setAttribute("thickness", 0.25f);
        iteration.setAttribute("n_tot", 5.0f);
        iteration.setAttribute("crystal_t_fluo", 1.25f);
        iteration.setAttribute("cladding_number", 7u);
        iteration.setAttribute("cladding_absorption", 0.05);
        iteration.setAttribute("min_rays_per_sample", 1u);
        iteration.setAttribute("max_rays_per_sample", 2u);
        iteration.setAttribute("mse_threshold", 0.5);
        iteration.setAttribute("repetitions", 3u);
        iteration.setAttribute("adaptive_steps", 4u);
        iteration.setAttribute("max_gpus", 1u);
        iteration.setAttribute("backend", std::string{"Host_Cpu_CpuSerial"});
        iteration.setAttribute("parallel_mode", std::string{"single"});
        iteration.setAttribute("min_sample_range", 0u);
        iteration.setAttribute("max_sample_range", 5u);
        iteration.setAttribute("rng_seed", 1234u);
        iteration.setAttribute("use_reflections", true);
        iteration.setAttribute("spectral_resolution", 2u);
        iteration.setAttribute("monochromatic", false);
        iteration.setAttribute("max_sigma_absorption", 0.02);
        iteration.setAttribute("max_sigma_emission", 0.04);

        writeComponent<double>(series, iteration, "core_vertices", "x", {0.0, 1.0, 0.0}, {"point"});
        writeComponent<double>(series, iteration, "core_vertices", "y", {0.0, 0.0, 1.0}, {"point"});
        setMetadata(iteration.meshes["core_vertices"], {"coordinate", "point"}, {2u, 3u}, false, true, "m", {"point"});
        writeComponent<double>(series, iteration, "core_cell_center", "x", {1.0 / 3.0}, {"cell"});
        writeComponent<double>(series, iteration, "core_cell_center", "y", {1.0 / 3.0}, {"cell"});
        setMetadata(iteration.meshes["core_cell_center"], {"cell"}, {1u}, false, true, "m", {"cell"});

        writeScalar<unsigned>(
            series,
            iteration,
            "core_connectivity",
            {0u, 1u, 2u},
            {"cell", "local_vertex"},
            {1u, 3u});
        writeScalar<int>(series, iteration, "core_neighbors", {-1, -1, -1}, {"cell", "local_side"}, {1u, 3u});
        writeScalar<int>(series, iteration, "core_forbidden_edges", {-1, -1, -1}, {"cell", "local_side"}, {1u, 3u});
        writeScalar<unsigned>(series, iteration, "core_normal_points", {0u, 1u, 2u}, {"cell", "local_side"}, {1u, 3u});
        writeScalar<double>(
            series,
            iteration,
            "core_cell_normal_x",
            {0.0, 1.0, -1.0},
            {"cell", "local_side"},
            {1u, 3u});
        writeScalar<double>(
            series,
            iteration,
            "core_cell_normal_y",
            {-1.0, 1.0, 0.0},
            {"cell", "local_side"},
            {1u, 3u});
        writeScalar<float>(series, iteration, "core_surface", {0.5f}, {"cell"}, {1u}, false, true, "m^2");
        if(betaVolumeAsFloat)
        {
            writeScalar<float>(series, iteration, "core_beta_volume", {0.1f}, {"cell", "layer"}, {1u, 1u}, true);
        }
        else
        {
            writeScalar<double>(series, iteration, "core_beta_volume", {0.1}, {"cell", "layer"}, {1u, 1u}, true);
        }
        if(pointBetaTransposed)
        {
            auto record = iteration.meshes["core_point_beta"];
            setMetadata(record, {"point", "level"}, {3u, 2u}, true);
            auto& component = record[io::MeshRecordComponent::SCALAR];
            std::vector<double> values{0.1, 0.4, 0.2, 0.5, 0.3, 0.6};
            io::Extent extent{2u, 3u};
            component.setUnitSI(1.0);
            component.setPosition(std::vector<double>{0.0, 0.0});
            component.resetDataset({io::determineDatatype<double>(), extent});
            component.storeChunk(values, io::Offset{0u, 0u}, extent);
            series.flush();
        }
        else
        {
            writeScalar<double>(
                series,
                iteration,
                "core_point_beta",
                {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
                {"point", "level"},
                {3u, 2u},
                true);
        }
        writeScalar<unsigned>(series, iteration, "core_cladding_cell_type", {0u}, {"cell"}, {1u});
        writeScalar<float>(series, iteration, "core_refractive_index", {1.5f, 1.0f, 1.5f, 1.0f}, {"interface"}, {4u});
        writeScalar<float>(series, iteration, "core_reflectivity", {0.1f, 0.2f}, {"cell", "interface"}, {1u, 2u});
        writeScalar<double>(
            series,
            iteration,
            "core_lambda_absorption",
            {900.0, 910.0},
            {"wavelength"},
            {2u},
            false,
            false,
            "m");
        writeScalar<double>(
            series,
            iteration,
            "core_lambda_emission",
            {1000.0, 1010.0},
            {"wavelength"},
            {2u},
            false,
            false,
            "m");
        writeScalar<double>(
            series,
            iteration,
            "core_sigma_absorption",
            {0.01, 0.02},
            {"wavelength"},
            {2u},
            false,
            false,
            "cm^2");
        writeScalar<double>(
            series,
            iteration,
            "core_sigma_emission",
            {0.03, 0.04},
            {"wavelength"},
            {2u},
            false,
            false,
            "cm^2");

        if(mutate)
        {
            mutate(series, iteration);
        }

        iteration.close();
        series.close();
        return path;
    }

    std::string parserError(std::filesystem::path const& path)
    {
        hase::openpmd::Parser parser{path, testPath("unused-output")};
        try
        {
            (void) parser.read();
        }
        catch(std::runtime_error const& err)
        {
            return err.what();
        }
        return {};
    }
} // namespace

TEST_CASE("openPMD parser reads a schema-valid transport record", "[openpmd][parser]")
{
    auto const path = writeInput("valid");
    hase::openpmd::Parser parser{path, testPath("valid-output")};
    auto context = parser.read();

    REQUIRE(context.mesh.numberOfPoints == 3u);
    REQUIRE(context.mesh.numberOfTriangles == 1u);
    REQUIRE(context.mesh.numberOfLevels == 2u);
    REQUIRE(context.mesh.trianglePointIndices == std::vector<unsigned>{0u, 1u, 2u});
    REQUIRE(context.mesh.betaCells.size() == 6u);
    REQUIRE(context.experiment.spectral == 2u);
    REQUIRE(context.compute.maxRepetitions == 3u);
    REQUIRE(context.compute.writeVtk == false);
    REQUIRE(context.compute.devices.empty());
    REQUIRE(context.compute.rngSeed == 1234u);
}

TEST_CASE("openPMD parser rejects malformed fields before HostMesh construction", "[openpmd][parser]")
{
    SECTION("extent")
    {
        auto const path = writeInput(
            "bad_extent",
            [](io::Series& series, io::Iteration& iteration)
            {
                writeScalar<double>(
                    series,
                    iteration,
                    "core_beta_volume",
                    {0.1, 0.2},
                    {"cell", "layer"},
                    {1u, 1u},
                    true);
            });
        auto const error = parserError(path);
        REQUIRE(error.find("openPMD validation error for 'core_beta_volume'") != std::string::npos);
        REQUIRE(error.find("extent mismatch") != std::string::npos);
    }

    SECTION("accidental transpose extent with matching element count")
    {
        auto const path = writeInput("bad_transpose_extent", {}, false, true);
        auto const error = parserError(path);
        REQUIRE(error.find("openPMD validation error for 'core_point_beta'") != std::string::npos);
        REQUIRE(error.find("extent mismatch") != std::string::npos);
    }

    SECTION("dtype")
    {
        auto const path = writeInput("bad_dtype", {}, true);
        auto const error = parserError(path);
        REQUIRE(error.find("openPMD validation error for 'core_beta_volume'") != std::string::npos);
        REQUIRE(error.find("dtype mismatch") != std::string::npos);
    }

    SECTION("metadata")
    {
        auto const path = writeInput(
            "bad_metadata",
            [](io::Series& series, io::Iteration& iteration)
            {
                (void) series;
                iteration.meshes["core_connectivity"].setAttribute("haseSchemaVersion", std::string{"999"});
            });
        auto const error = parserError(path);
        REQUIRE(error.find("openPMD validation error for 'core_connectivity'") != std::string::npos);
        REQUIRE(error.find("haseSchemaVersion") != std::string::npos);
    }

    SECTION("role")
    {
        auto const path = writeInput(
            "bad_role",
            [](io::Series& series, io::Iteration& iteration)
            {
                (void) series;
                iteration.meshes["core_beta_volume"].setAttribute("haseDynamic", false);
            });
        auto const error = parserError(path);
        REQUIRE(error.find("openPMD validation error for 'core_beta_volume'") != std::string::npos);
        REQUIRE(error.find("haseDynamic") != std::string::npos);
    }
}

TEST_CASE("openPMD parser rejects unsupported compute settings explicitly", "[openpmd][parser]")
{
    auto const path = writeInput(
        "bad_compute",
        [](io::Series& series, io::Iteration& iteration)
        {
            (void) series;
            iteration.setAttribute("write_vtk", true);
        });
    auto const error = parserError(path);
    REQUIRE(error.find("openPMD validation error for 'compute/write_vtk'") != std::string::npos);
    REQUIRE(error.find("unsupported compute setting") != std::string::npos);
}

TEST_CASE("openPMD parser round-trips a Python writer contract input", "[openpmd][parser][python]")
{
    char const* input = std::getenv("HASE_OPENPMD_PYTHON_CONTRACT_INPUT");
    if(input == nullptr)
    {
        SKIP("HASE_OPENPMD_PYTHON_CONTRACT_INPUT is not set");
    }

    char const* outputEnv = std::getenv("HASE_OPENPMD_PYTHON_CONTRACT_OUTPUT");
    auto const output = outputEnv == nullptr ? testPath("python_contract_result") : std::filesystem::path{outputEnv};
    std::filesystem::remove_all(output);

    hase::openpmd::Parser parser{std::filesystem::path{input}, output};
    auto context = parser.read();

    REQUIRE(context.mesh.numberOfTriangles == 3u);
    REQUIRE(context.mesh.numberOfLevels == 6u);
    REQUIRE(context.mesh.numberOfPoints == 5u);
    REQUIRE(context.mesh.thickness == 0.375f);
    REQUIRE(context.mesh.trianglePointIndices == std::vector<unsigned>{0u, 2u, 0u, 1u, 3u, 2u, 2u, 4u, 4u});
    REQUIRE(context.mesh.triangleNeighbors == std::vector<int>{-1, -1, 0, -1, -1, 1, 2, 2, -1});
    REQUIRE(context.mesh.forbiddenEdge == std::vector<int>{-1, -1, 2, -1, -1, 2, 0, 1, -1});
    REQUIRE(context.mesh.triangleNormalPoint == std::vector<unsigned>{0u, 2u, 0u, 1u, 3u, 2u, 2u, 4u, 4u});
    REQUIRE(context.mesh.points == std::vector<double>{0.0, 1.5, 0.25, 2.25, -0.75, 0.0, -0.5, 1.25, 2.5, 3.75});
    REQUIRE(
        context.mesh.triangleCenterX
        == std::vector<double>{0.5833333333333333, 0.5833333333333333, -0.16666666666666666});
    REQUIRE(context.mesh.triangleCenterY == std::vector<double>{0.25, 2.5, 1.6666666666666665});
    REQUIRE(
        context.mesh.betaVolume
        == std::vector<
            double>{0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33, 0.14, 0.24, 0.34, 0.15, 0.25, 0.35});
    REQUIRE(context.mesh.betaCells == std::vector<double>{100.0, 110.0, 120.0, 130.0, 140.0, 101.0, 111.0, 121.0,
                                                          131.0, 141.0, 102.0, 112.0, 122.0, 132.0, 142.0, 103.0,
                                                          113.0, 123.0, 133.0, 143.0, 104.0, 114.0, 124.0, 134.0,
                                                          144.0, 105.0, 115.0, 125.0, 135.0, 145.0});
    REQUIRE(context.mesh.claddingCellTypes == std::vector<unsigned>{0u, 2u, 1u});
    REQUIRE(context.mesh.refractiveIndices == std::vector<float>{1.80f, 1.20f, 1.65f, 1.05f});
    REQUIRE(context.mesh.reflectivities == std::vector<float>{0.01f, 0.03f, 0.05f, 0.02f, 0.04f, 0.06f});
    REQUIRE(context.experiment.spectral == 3u);
    REQUIRE(context.experiment.lambdaA == std::vector<double>{900e-9, 910e-9, 930e-9});
    REQUIRE(context.experiment.lambdaE == std::vector<double>{1000e-9, 1015e-9, 1040e-9});
    REQUIRE(context.experiment.sigmaA == std::vector<double>{0.010, 0.025, 0.040});
    REQUIRE(context.experiment.sigmaE == std::vector<double>{0.050, 0.035, 0.020});
    REQUIRE(context.compute.maxRepetitions == 1u);
    REQUIRE(context.compute.minSampleRange == 0u);
    REQUIRE(context.compute.maxSampleRange == 0u);
    REQUIRE(context.compute.rngSeed == 1234u);

    std::vector<float> phiAse(30u);
    std::vector<double> mse(30u);
    std::vector<unsigned> totalRays(30u);
    std::vector<double> dndtAse(30u);
    for(unsigned i = 0; i < 30u; ++i)
    {
        phiAse[i] = 0.5f + static_cast<float>(i);
        mse[i] = 1000.0 + static_cast<double>(i);
        totalRays[i] = 200u + i;
        dndtAse[i] = -10.0 - static_cast<double>(i);
    }

    parser.writeResult(hase::core::Result{phiAse, mse, totalRays, dndtAse}, context.mesh);
}
