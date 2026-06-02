#pragma once

#include <core/mesh.hpp>
#include <core/types.hpp>

#include <filesystem>
#include <string>

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
#    include <mpi.h>
#endif

namespace hase::openpmd
{

    struct SimulationContext
    {
        core::ExperimentParameters experiment;
        core::ComputeParameters compute;
        core::HostMesh mesh;
        core::Result result;
    };

    class Parser
    {
    public:
        Parser(std::filesystem::path inputPath, std::filesystem::path outputPath);

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        Parser(std::filesystem::path inputPath, std::filesystem::path outputPath, MPI_Comm comm);
#endif

        [[nodiscard]] SimulationContext read();
        void writeResult(core::Result const& result, core::HostMesh const& mesh);

    private:
        [[nodiscard]] bool isHeadRank() const;

        std::filesystem::path m_inputPath;
        std::filesystem::path m_outputPath;
        std::string m_meshGroup = "core";

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        MPI_Comm m_comm = MPI_COMM_WORLD;
#endif
    };

} // namespace hase::openpmd
