#pragma once

#include <core/simulationSnapshot.hpp>
#include <core/types.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
#    include <mpi.h>
#endif

namespace hase::openpmd
{
    class OutputWriter
    {
    public:
        explicit OutputWriter(std::filesystem::path outputPath);

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        OutputWriter(std::filesystem::path outputPath, MPI_Comm comm);
#endif

        OutputWriter(OutputWriter&&) noexcept;
        OutputWriter& operator=(OutputWriter&&) noexcept;
        OutputWriter(OutputWriter const&) = delete;
        OutputWriter& operator=(OutputWriter const&) = delete;
        ~OutputWriter();

        void writeResult(std::uint64_t iterationIndex, core::Result const& result);
        void writeSnapshot(std::uint64_t iterationIndex, core::SimulationSnapshot const& snapshot);
        void close();

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
} // namespace hase::openpmd
