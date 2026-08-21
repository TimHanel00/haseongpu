#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <data/Simulation.hpp>
#include <data/SimulationPreparation.hpp>
#include <data/TimeIntegrationSolver.hpp>
#include <openPMD/openPMD.hpp>
#include <transport/TransportReader.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace io = openPMD;

namespace
{
    std::filesystem::path transportPath()
    {
        auto path = std::filesystem::temp_directory_path() / "hase_transport_reader.bp";
        std::filesystem::remove_all(path);
        return path;
    }

    template<typename T>
    void writeScalar(io::Series& series, io::Iteration& iteration, std::string const& field, T value)
    {
        auto record = iteration.meshes["hase__" + field];
        record.setAttribute("hasePath", std::string{"timeIntegrationSolver/"} + field);
        auto& component = record[io::MeshRecordComponent::SCALAR];
        component.setUnitSI(1.0);
        component.resetDataset(io::Dataset{io::determineDatatype<T>(), io::Extent{1u}});
        std::vector<T> values{value};
        component.storeChunk(values, io::Offset{0u}, io::Extent{1u});
        series.flush();
    }

    template<typename T>
    void writeArray(
        io::Series& series,
        io::Iteration& iteration,
        std::string const& owner,
        std::string const& field,
        std::vector<T> values)
    {
        auto record = iteration.meshes["hase__" + field];
        record.setAttribute("hasePath", owner + "/" + field);
        auto& component = record[io::MeshRecordComponent::SCALAR];
        component.setUnitSI(1.0);
        component.resetDataset(io::Dataset{io::determineDatatype<T>(), io::Extent{values.size()}});
        component.storeChunk(values, io::Offset{0u}, io::Extent{values.size()});
        series.flush();
    }
} // namespace

TEST_CASE("transport reader delegates a subtree to its primitive", "[transport]")
{
    auto const path = transportPath();
    {
        io::Series series(path.string(), io::Access::CREATE);
        auto iteration = series.snapshots()[0u];
        iteration.setAttribute("haseTransportVersion", std::string{"1.2"});
        iteration.setAttribute("haseUpdateMode", std::string{"full"});
        iteration.setAttribute("haseRoot", std::string{"timeIntegrationSolver"});
        iteration.setAttribute("haseNodePaths", std::vector<std::string>{"timeIntegrationSolver"});
        iteration.setAttribute("haseNodeTypes", std::vector<std::string>{"timeIntegrationSolver"});
        iteration.setAttribute("hase__attribute__timeIntegrationSolver%2Fname", std::string{"implicit-éuler\n🚀"});
        iteration.setAttribute("hase__attribute__timeIntegrationSolver%2Flabels", std::vector<std::string>{"α", "β"});
        iteration.setAttribute("hase__attribute__timeIntegrationSolver%2FemptyLabels", std::string{"[]"});
        iteration.setAttribute("hase__reference__timeIntegrationSolver%2FemptyReferences", std::string{"[]"});
        writeScalar<std::uint64_t>(series, iteration, "iterations", 7u);
        writeScalar<double>(series, iteration, "tolerance", 1.0e-8);
        iteration.close();
        series.close();
    }

    {
        io::Series series(path.string(), io::Access::READ_ONLY);
        auto iteration = series.snapshots().begin()->second;
        hase::transport::TransportReader reader(series, iteration);
        auto const solver = hase::data::TimeIntegrationSolver::fromTransport(reader, reader.root());
        CHECK(solver.name == "implicit-éuler\n🚀");
        REQUIRE(solver.iterations);
        CHECK(*solver.iterations == 7u);
        REQUIRE(solver.tolerance);
        CHECK(*solver.tolerance == Catch::Approx(1.0e-8));
        std::vector<std::string> labels;
        reader.assign(labels, reader.root(), "labels");
        CHECK(labels == std::vector<std::string>{"α", "β"});
        std::vector<std::string> emptyLabels{"must be replaced"};
        reader.assign(emptyLabels, reader.root(), "emptyLabels");
        CHECK(emptyLabels.empty());
        CHECK(reader.referencePaths(reader.root().child("emptyReferences").string()).empty());
        series.close();
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("transport reader preserves uint64 values exactly", "[transport]")
{
    auto const path = transportPath();
    auto const expected = std::numeric_limits<std::uint64_t>::max() - 2u;
    {
        io::Series series(path.string(), io::Access::CREATE);
        auto iteration = series.snapshots()[0u];
        iteration.setAttribute("haseTransportVersion", std::string{"1.2"});
        iteration.setAttribute("haseUpdateMode", std::string{"full"});
        iteration.setAttribute("haseRoot", std::string{"timeIntegrationSolver"});
        iteration.setAttribute("haseNodePaths", std::vector<std::string>{"timeIntegrationSolver"});
        iteration.setAttribute("haseNodeTypes", std::vector<std::string>{"timeIntegrationSolver"});
        iteration.setAttribute("hase__attribute__timeIntegrationSolver%2Fname", std::string{"implicit-euler"});
        writeScalar<std::uint64_t>(series, iteration, "iterations", expected);
        iteration.close();
        series.close();
    }

    {
        io::Series series(path.string(), io::Access::READ_ONLY);
        auto iteration = series.snapshots().begin()->second;
        hase::transport::TransportReader reader(series, iteration);
        auto const solver = hase::data::TimeIntegrationSolver::fromTransport(reader, reader.root());
        REQUIRE(solver.iterations);
        CHECK(*solver.iterations == expected);
        series.close();
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("dynamic transport explicitly replaces a resized cross-section table", "[transport][material][update]")
{
    auto const path = transportPath();
    {
        io::Series series(path.string(), io::Access::CREATE);
        auto iteration = series.snapshots()[1u];
        iteration.setAttribute("haseTransportVersion", std::string{"1.2"});
        iteration.setAttribute("haseUpdateMode", std::string{"dynamic"});
        iteration.setAttribute("haseRoot", std::string{"crossSectionTable"});
        iteration.setAttribute("haseNodePaths", std::vector<std::string>{"crossSectionTable"});
        iteration.setAttribute("haseNodeTypes", std::vector<std::string>{"crossSectionTable"});
        writeArray(
            series,
            iteration,
            "crossSectionTable",
            "wavelengths",
            std::vector<double>{900e-9, 950e-9, 1000e-9});
        writeArray(
            series,
            iteration,
            "crossSectionTable",
            "absorption",
            std::vector<double>{1.0e-25, 2.0e-25, 3.0e-25});
        writeArray(series, iteration, "crossSectionTable", "emission", std::vector<double>{4.0e-25, 5.0e-25, 6.0e-25});
        iteration.close();
        series.close();
    }

    {
        io::Series series(path.string(), io::Access::READ_ONLY);
        auto iteration = series.snapshots().begin()->second;
        hase::transport::TransportReader reader(series, iteration);
        hase::data::CrossSectionTable table;
        table.replaceSamples({1030e-9}, {7.0e-25}, {8.0e-25});
        table.updateFromTransport(reader, reader.root());
        CHECK(table.wavelengths.values == std::vector<double>{900e-9, 950e-9, 1000e-9});
        CHECK(table.absorption.values == std::vector<double>{1.0e-25, 2.0e-25, 3.0e-25});
        CHECK(table.emission.values == std::vector<double>{4.0e-25, 5.0e-25, 6.0e-25});
        CHECK(table.wavelengths.shape == std::vector<std::uint64_t>{3u});
        CHECK(table.absorption.shape == std::vector<std::uint64_t>{3u});
        CHECK(table.emission.shape == std::vector<std::uint64_t>{3u});
        series.close();
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("frontend graph parses through Simulation into trace preparation", "[transport][integration]")
{
    auto const* input = std::getenv("HASE_TEST_TRANSPORT_GRAPH");
    if(input == nullptr)
        SKIP("HASE_TEST_TRANSPORT_GRAPH is not set");

    io::Series series(input, io::Access::READ_ONLY);
    auto iteration = series.snapshots().begin()->second;
    hase::transport::TransportReader reader(series, iteration);
    auto const componentPaths = reader.referencePaths(reader.root().child("opticalComponents").string());
    REQUIRE(componentPaths.size() == 1u);
    auto const domainPaths
        = reader.referencePaths(hase::transport::TransportPath{componentPaths.front()}.child("domain").string());
    REQUIRE(domainPaths.size() == 1u);
    auto const domain = hase::data::Domain::fromTransport(reader, hase::transport::TransportPath{domainPaths.front()});
    REQUIRE(domain.topologies.size() == 1u);
    CHECK(domain.topologies.front()->points.shape == std::vector<std::uint64_t>{3u, 4u});
    auto const simulation = hase::data::Simulation::fromTransport(reader, reader.root());
    auto state = hase::data::prepareSimulation(simulation);
    CHECK(state.trace.numberOfCells == 1u);
    CHECK(state.trace.numberOfMeshPoints == 4u);
    CHECK(state.trace.materialActiveIonDensities.front() == Catch::Approx(2.76e26));
    CHECK(state.trace.crossSectionEmission.front() == Catch::Approx(2.1e-24));
    CHECK(state.controls.numberOfSteps == 1u);
    series.close();
}
