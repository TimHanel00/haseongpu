#pragma once

#include <core/Runtime.hpp>
#include <data/SimulationSnapshot.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
#    include <mpi.h>
#endif

namespace hase::openpmd
{
    /** @brief Move-only writer for result and simulation-snapshot iterations. */
    class OutputWriter
    {
    public:
        /**
         * @param outputPath openPMD series path or streaming endpoint.
         */
        explicit OutputWriter(std::filesystem::path outputPath);

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        /**
         * @param outputPath openPMD series path or streaming endpoint.
         * @param comm MPI communicator collectively owning the series.
         */
        OutputWriter(std::filesystem::path outputPath, MPI_Comm comm);
#endif

        OutputWriter(OutputWriter&&) noexcept;
        OutputWriter& operator=(OutputWriter&&) noexcept;
        OutputWriter(OutputWriter const&) = delete;
        OutputWriter& operator=(OutputWriter const&) = delete;
        ~OutputWriter();

        /**
         * @brief Write and flush one standalone PhiASE result iteration.
         * @param iterationIndex openPMD iteration index to create.
         * @param result Cell-ordered ASE result and diagnostics.
         */
        void writeResult(std::uint64_t iterationIndex, data::PhiAseResult const& result);

        /**
         * @brief Write and flush the fields selected in one simulation snapshot.
         * @param iterationIndex openPMD iteration index to create.
         * @param snapshot Completed simulation step and selected output fields.
         */
        void writeSnapshot(std::uint64_t iterationIndex, data::SimulationSnapshot const& snapshot);

        /** @brief Close the openPMD series; repeated calls are harmless. */
        void close();

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
} // namespace hase::openpmd
