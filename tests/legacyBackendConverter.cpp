#include <backend/legacy/LegacyBackendConverter.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>

TEST_CASE("legacy converter lowers the primitive graph at one boundary", "[transport][legacy]")
{
    using namespace hase::backend;

    auto topology = std::make_shared<VolumeTopology>();
    topology->points.values = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    topology->cellPointIndices.values = {0u, 1u, 2u, 3u};
    topology->cellTypes.values = {10u};
    topology->cellDomains.values = {1};
    topology->facePointIndices.values = {0, 0, 1, 2, 2, 1, 2, 0, 1, 3, 3, 3};
    topology->neighborCells.values = {-1, -1, -1, -1};
    topology->neighborLocalFaces.values = {-1, -1, -1, -1};
    topology->faceBoundaries.values = {-1, -1, -1, -1};
    topology->cellCenters.values = {0.25, 0.25, 0.25};
    topology->cellVolumes.values = {1.0 / 6.0};

    auto volume = std::make_shared<Domain>();
    volume->entityKind = "volume";
    volume->masks = {{1u}, {0u, 1u}};
    volume->topologies = {topology};

    auto table = std::make_shared<CrossSectionTable>();
    table->wavelengths.values = {940.0e-9};
    table->absorption.values = {1.2e-25};
    table->emission.values = {2.1e-24};

    auto material = std::make_shared<Material>();
    material->materialName = "test gain material";
    material->active = true;
    material->refractiveIndex = 1.8;
    material->fluorescenceLifetime = 9.5e-4;
    material->activeIonDensity = 2.76e26;
    material->crossSections = table;

    auto component = std::make_shared<OpticalComponent>();
    component->domain = volume;
    component->material = material;

    auto gainMedium = std::make_shared<GainMedium>();
    gainMedium->components = {component};

    auto excitation = std::make_shared<ExcitationState>();
    excitation->domains = {volume};
    excitation->values = {{0.25}, {0u, 1u}};

    auto phiAse = std::make_shared<PhiAse>();
    phiAse->propagationMode = "forward";
    phiAse->minRays = 10u;
    phiAse->maxRays = 20u;
    phiAse->relativeStandardErrorThreshold = 0.1;
    phiAse->repetitions = 2u;
    phiAse->adaptiveSteps = 3u;
    phiAse->reflectionMaxIterations = 40u;
    phiAse->reflectionTolerance = 1.0e-4;
    phiAse->surfaceReservoirSize = 32u;
    phiAse->backend = "test-backend";
    phiAse->parallelMode = "single";
    phiAse->numDevices = 1u;
    phiAse->devices = hase::backend::transport::Array<std::uint64_t>{};

    auto solver = std::make_shared<TimeIntegrationSolver>();
    solver->name = "explicit-euler";

    Simulation simulation;
    simulation.timeStep = 1.0e-6;
    simulation.simulationSteps = 2u;
    simulation.executionMode = "autonomous";
    simulation.outputFields = {"beta_volume", "phi_ase"};
    simulation.opticalComponents = {component};
    simulation.gainMedium = gainMedium;
    simulation.exteriorSurface = std::make_shared<Domain>();
    simulation.excitationState = excitation;
    simulation.phiAse = phiAse;
    simulation.timeIntegrationSolver = solver;

    auto converted = hase::backend::legacy::LegacyBackendConverter::convert(simulation);

    CHECK(converted.mesh.numberOfCells == 1u);
    CHECK(converted.mesh.numberOfMeshPoints == 4u);
    CHECK(converted.mesh.betaVolume == std::vector<double>{0.25});
    CHECK(converted.mesh.nTot == Catch::Approx(2.76e20));
    CHECK(converted.mesh.crystalTFluo == Catch::Approx(9.5e-4));
    CHECK(converted.experiment.sigmaA.front() == Catch::Approx(1.2e-21));
    CHECK(converted.experiment.sigmaE.front() == Catch::Approx(2.1e-20));
    CHECK(converted.run.numberOfSteps == 2u);
    CHECK(converted.run.timeIntegration.method == "explicit-euler");
}

TEST_CASE("legacy converter assembles independent domain-local topology shards", "[transport][legacy][partition]")
{
    using namespace hase::backend;

    auto makeTopology = [](double offset)
    {
        auto topology = std::make_shared<VolumeTopology>();
        topology->points.values = {offset, offset + 1.0, offset, offset, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        topology->cellPointIndices.values = {0u, 1u, 2u, 3u};
        topology->cellTypes.values = {10u};
        topology->cellDomains.values = {1};
        topology->facePointIndices.values = {0, 0, 1, 2, 2, 1, 2, 0, 1, 3, 3, 3};
        topology->neighborCells.values = {-1, -1, -1, -1};
        topology->neighborLocalFaces.values = {-1, -1, -1, -1};
        topology->faceBoundaries.values = {-1, -1, -1, -1};
        topology->cellCenters.values = {offset + 0.25, 0.25, 0.25};
        topology->cellVolumes.values = {1.0 / 6.0};
        return topology;
    };
    auto makeDomain = [](std::shared_ptr<VolumeTopology> const& topology)
    {
        auto domain = std::make_shared<Domain>();
        domain->entityKind = "volume";
        domain->masks = {{1u}, {0u, 1u}};
        domain->topologies = {topology};
        return domain;
    };

    auto table = std::make_shared<CrossSectionTable>();
    table->wavelengths.values = {1030.0e-9};
    table->absorption.values = {1.0e-25};
    table->emission.values = {2.0e-24};
    auto material = std::make_shared<Material>();
    material->materialName = "shared gain material";
    material->active = true;
    material->refractiveIndex = 1.8;
    material->fluorescenceLifetime = 1.0e-3;
    material->activeIonDensity = 2.76e26;
    material->crossSections = table;

    std::vector<std::shared_ptr<OpticalComponent>> components;
    std::vector<std::shared_ptr<Domain>> domains;
    for(double offset : {0.0, 10.0})
    {
        auto domain = makeDomain(makeTopology(offset));
        auto component = std::make_shared<OpticalComponent>();
        component->domain = domain;
        component->material = material;
        domains.push_back(domain);
        components.push_back(component);
    }

    auto gainMedium = std::make_shared<GainMedium>();
    gainMedium->components = components;
    auto excitation = std::make_shared<ExcitationState>();
    excitation->domains = domains;
    excitation->values = {{0.2, 0.4}, {0u, 1u, 2u}};
    auto phiAse = std::make_shared<PhiAse>();
    phiAse->propagationMode = "forward";
    phiAse->minRays = 10u;
    phiAse->maxRays = 10u;
    phiAse->relativeStandardErrorThreshold = 0.1;
    phiAse->repetitions = 1u;
    phiAse->adaptiveSteps = 0u;
    phiAse->reflectionMaxIterations = 40u;
    phiAse->reflectionTolerance = 1.0e-4;
    phiAse->surfaceReservoirSize = 32u;
    phiAse->backend = "test-backend";
    phiAse->parallelMode = "single";
    phiAse->numDevices = 1u;
    phiAse->devices = hase::backend::transport::Array<std::uint64_t>{};
    auto solver = std::make_shared<TimeIntegrationSolver>();
    solver->name = "explicit-euler";

    Simulation simulation;
    simulation.timeStep = 1.0e-6;
    simulation.simulationSteps = 1u;
    simulation.executionMode = "autonomous";
    simulation.outputFields = {"beta_volume"};
    simulation.opticalComponents = components;
    simulation.gainMedium = gainMedium;
    simulation.exteriorSurface = std::make_shared<Domain>();
    simulation.excitationState = excitation;
    simulation.phiAse = phiAse;
    simulation.timeIntegrationSolver = solver;

    auto converted = hase::backend::legacy::LegacyBackendConverter::convert(simulation);
    CHECK(converted.mesh.numberOfCells == 2u);
    CHECK(converted.mesh.numberOfMeshPoints == 8u);
    CHECK(converted.mesh.betaVolume == std::vector<double>{0.2, 0.4});
    CHECK(converted.mesh.cellPointIndices == std::vector<unsigned>{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u});
    CHECK(converted.mesh.cellCenters == std::vector<double>{0.25, 10.25, 0.25, 0.25, 0.25, 0.25});
}
