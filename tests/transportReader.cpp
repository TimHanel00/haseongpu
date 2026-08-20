#include <backend/legacy/LegacyBackendConverter.hpp>
#include <backend/primitives/Simulation.hpp>
#include <backend/primitives/TimeIntegrationSolver.hpp>
#include <backend/transport/TransportReader.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <openPMD/openPMD.hpp>

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
} // namespace

TEST_CASE("transport reader delegates a subtree to its primitive", "[transport]")
{
    auto const path = transportPath();
    {
        io::Series series(path.string(), io::Access::CREATE);
        auto iteration = series.snapshots()[0u];
        iteration.setAttribute("haseTransportVersion", std::string{"1.1"});
        iteration.setAttribute("haseUpdateMode", std::string{"full"});
        iteration.setAttribute("haseRoot", std::string{"timeIntegrationSolver"});
        iteration.setAttribute("haseNodePaths", std::vector<std::string>{"timeIntegrationSolver"});
        iteration.setAttribute("haseNodeTypes", std::vector<std::string>{"timeIntegrationSolver"});
        iteration.setAttribute("hase__attribute__timeIntegrationSolver%2Fname", std::string{"implicit-éuler\n🚀"});
        iteration.setAttribute("hase__attribute__timeIntegrationSolver%2Flabels", std::vector<std::string>{"α", "β"});
        writeScalar<std::uint64_t>(series, iteration, "iterations", 7u);
        writeScalar<double>(series, iteration, "tolerance", 1.0e-8);
        iteration.close();
        series.close();
    }

    {
        io::Series series(path.string(), io::Access::READ_ONLY);
        auto iteration = series.snapshots().begin()->second;
        hase::backend::transport::TransportReader reader(series, iteration);
        auto const solver = hase::backend::TimeIntegrationSolver::fromTransport(reader, reader.root());
        CHECK(solver.name == "implicit-éuler\n🚀");
        REQUIRE(solver.iterations);
        CHECK(*solver.iterations == 7u);
        REQUIRE(solver.tolerance);
        CHECK(*solver.tolerance == Catch::Approx(1.0e-8));
        std::vector<std::string> labels;
        reader.assign(labels, reader.root(), "labels");
        CHECK(labels == std::vector<std::string>{"α", "β"});
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
        iteration.setAttribute("haseTransportVersion", std::string{"1.1"});
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
        hase::backend::transport::TransportReader reader(series, iteration);
        auto const solver = hase::backend::TimeIntegrationSolver::fromTransport(reader, reader.root());
        REQUIRE(solver.iterations);
        CHECK(*solver.iterations == expected);
        series.close();
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("frontend graph parses through Simulation into the legacy boundary", "[transport][integration]")
{
    auto const* input = std::getenv("HASE_TEST_TRANSPORT_GRAPH");
    if(input == nullptr)
        SKIP("HASE_TEST_TRANSPORT_GRAPH is not set");

    io::Series series(input, io::Access::READ_ONLY);
    auto iteration = series.snapshots().begin()->second;
    hase::backend::transport::TransportReader reader(series, iteration);
    auto const componentPaths = reader.referencePaths(reader.root().child("opticalComponents").string());
    REQUIRE(componentPaths.size() == 1u);
    auto const domainPaths = reader.referencePaths(
        hase::backend::transport::TransportPath{componentPaths.front()}.child("domain").string());
    REQUIRE(domainPaths.size() == 1u);
    auto const domain
        = hase::backend::Domain::fromTransport(reader, hase::backend::transport::TransportPath{domainPaths.front()});
    REQUIRE(domain.topologies.size() == 1u);
    CHECK(domain.topologies.front()->points.shape == std::vector<std::uint64_t>{3u, 4u});
    auto const simulation = hase::backend::Simulation::fromTransport(reader, reader.root());
    auto converted = hase::backend::legacy::LegacyBackendConverter::convert(simulation);
    CHECK(converted.mesh.numberOfCells == 1u);
    CHECK(converted.mesh.numberOfMeshPoints == 4u);
    CHECK(converted.mesh.nTot == Catch::Approx(2.76e20));
    CHECK(converted.experiment.sigmaE.front() == Catch::Approx(2.1e-20));
    CHECK(converted.run.numberOfSteps == 1u);
    series.close();
}
