#pragma once

#include <data/Simulation.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
#    include <mpi.h>
#endif

namespace hase::openpmd
{
    /** @brief One indexed transport iteration reconstructed as a primitive graph. */
    struct TransportIteration
    {
        std::uint64_t index{};
        data::Simulation simulation;
    };

    /** @brief Move-only sequential session over an openPMD transport stream. */
    class InputSession
    {
    public:
        InputSession() = default;
        InputSession(InputSession&&) noexcept = default;
        InputSession& operator=(InputSession&&) noexcept = default;
        InputSession(InputSession const&) = delete;
        InputSession& operator=(InputSession const&) = delete;
        ~InputSession();

        /**
         * @param next Callback returning the next available transport iteration.
         * @param close Callback releasing the underlying openPMD series.
         */
        InputSession(std::function<std::optional<TransportIteration>()> next, std::function<void()> close);

        /**
         * @return Next reconstructed iteration, or `std::nullopt` after end of stream.
         * @throws std::runtime_error If the session is not initialized.
         */
        [[nodiscard]] std::optional<TransportIteration> next();

        /** @brief Close the underlying series; repeated calls are harmless. */
        void close();

    private:
        std::function<std::optional<TransportIteration>()> m_next;
        std::function<void()> m_close;
    };

    /** @brief Entry point for one openPMD primitive-graph source. */
    class Parser
    {
    public:
        /** @param inputPath openPMD series path or streaming endpoint. */
        explicit Parser(std::filesystem::path inputPath);

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        /**
         * @param inputPath openPMD series path or streaming endpoint.
         * @param comm MPI communicator used to open the series collectively.
         */
        Parser(std::filesystem::path inputPath, MPI_Comm comm);
#endif

        /**
         * @return Simulation graph from the first available iteration.
         * @throws std::runtime_error If the source provides no iteration.
         */
        [[nodiscard]] data::Simulation read() const;

        /** @return Sequential input session for all available iterations. */
        [[nodiscard]] InputSession open() const;

    private:
        std::filesystem::path m_inputPath;

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        MPI_Comm m_comm = MPI_COMM_WORLD;
#endif
    };
} // namespace hase::openpmd
