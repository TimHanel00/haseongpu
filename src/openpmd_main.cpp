#include <backend/legacy/LegacyBackendConverter.hpp>
#include <core/simulation.hpp>
#include <core/timeSteppedSimulation.hpp>
#include <openpmd/OpenPmdOutputWriter.hpp>
#include <openpmd/OpenPmdParser.hpp>
#include <openpmd/SimulationSnapshotWriter.hpp>

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace
{
    struct Paths
    {
        std::filesystem::path input;
        std::filesystem::path output;
        bool cppControl = false;
    };

    std::optional<std::string_view> valueFor(std::string_view arg, std::string_view name)
    {
        std::string const prefix = "--" + std::string(name) + "=";
        if(arg.starts_with(prefix))
            return arg.substr(prefix.size());
        return std::nullopt;
    }

    Paths parsePaths(int argc, char** argv)
    {
        Paths paths;
        for(int index = 1; index < argc; ++index)
        {
            std::string_view const argument = argv[index];
            if(auto value = valueFor(argument, "input-path"))
                paths.input = std::string(*value);
            else if(auto value = valueFor(argument, "output-path"))
                paths.output = std::string(*value);
            else if(argument == "--cpp-control")
                paths.cppControl = true;
            else
                throw std::runtime_error(
                    "Unsupported argument '" + std::string(argument)
                    + "'. calcPhiASE only accepts --input-path, --output-path, and --cpp-control.");
        }
        if(paths.input.empty())
            throw std::runtime_error("Missing required --input-path=<openPMD-series>.");
        if(paths.output.empty())
            throw std::runtime_error("Missing required --output-path=<openPMD-series>.");
        return paths;
    }

    bool isHeadRank()
    {
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank == 0;
#else
        return true;
#endif
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto const paths = parsePaths(argc, argv);
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        MPI_Init(&argc, &argv);
        hase::openpmd::Parser parser{paths.input, MPI_COMM_WORLD};
#else
        hase::openpmd::Parser parser{paths.input};
#endif
        auto input = parser.open();
        std::unique_ptr<hase::openpmd::OutputWriter> output;
        if(isHeadRank())
        {
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
            output = std::make_unique<hase::openpmd::OutputWriter>(paths.output, MPI_COMM_SELF);
#else
            output = std::make_unique<hase::openpmd::OutputWriter>(paths.output);
#endif
        }

        if(!paths.cppControl)
        {
            while(auto iteration = input.next())
            {
                auto legacyContext = hase::backend::legacy::LegacyBackendConverter::convert(iteration->simulation);
                int const status = hase::core::startSimulation<false>(
                    legacyContext.experiment,
                    legacyContext.compute,
                    legacyContext.result,
                    legacyContext.mesh);
                if(status != 0)
                    throw std::runtime_error("simulation failed with return code " + std::to_string(status));
                if(output)
                    output->writeResult(iteration->index, legacyContext.result);
            }
        }
        else
        {
            auto initial = input.next();
            if(!initial)
                throw std::runtime_error("No simulation iteration was available in the openPMD input stream.");
            auto legacyContext = hase::backend::legacy::LegacyBackendConverter::convert(initial->simulation);

            hase::openpmd::AsyncSimulationSnapshotWriter snapshots{
                output != nullptr,
                [&](hase::core::SimulationSnapshot const& snapshot)
                { output->writeSnapshot(snapshot.step - 1u, snapshot); },
                legacyContext.compute.parallelMode != hase::core::ParallelMode::MPI};

            int const status = hase::core::startTimeSteppedSimulation(
                legacyContext.experiment,
                legacyContext.compute,
                legacyContext.run,
                legacyContext.mesh,
                [&](hase::core::SimulationSnapshot const& snapshot) { snapshots.enqueue(snapshot); },
                [&](unsigned completedStep)
                {
                    auto update = input.next();
                    if(!update || update->index != completedStep)
                        throw std::runtime_error(
                            "synchronized-debug expected transport iteration " + std::to_string(completedStep));
                    auto converted = hase::backend::legacy::LegacyBackendConverter::convert(update->simulation);
                    return converted.mesh.betaVolume;
                });
            snapshots.finish();
            if(status != 0)
                throw std::runtime_error("simulation failed with return code " + std::to_string(status));
        }

        input.close();
        if(output)
            output->close();
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        MPI_Finalize();
#endif
        return 0;
    }
    catch(std::exception const& error)
    {
        std::cerr << "calcPhiASE failed: " << error.what() << '\n';
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if(initialized && !finalized)
            MPI_Finalize();
#endif
        return 1;
    }
}
