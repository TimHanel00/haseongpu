#pragma once

#include <backend/primitives/Simulation.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
#    include <mpi.h>
#endif

namespace hase::openpmd
{
    struct TransportIteration
    {
        std::uint64_t index{};
        backend::Simulation simulation;
    };

    class InputSession
    {
    public:
        InputSession() = default;
        InputSession(InputSession&&) noexcept = default;
        InputSession& operator=(InputSession&&) noexcept = default;
        InputSession(InputSession const&) = delete;
        InputSession& operator=(InputSession const&) = delete;
        ~InputSession();

        InputSession(std::function<std::optional<TransportIteration>()> next, std::function<void()> close);

        [[nodiscard]] std::optional<TransportIteration> next();
        void close();

    private:
        std::function<std::optional<TransportIteration>()> m_next;
        std::function<void()> m_close;
    };

    class Parser
    {
    public:
        explicit Parser(std::filesystem::path inputPath);

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        Parser(std::filesystem::path inputPath, MPI_Comm comm);
#endif

        [[nodiscard]] backend::Simulation read() const;
        [[nodiscard]] InputSession open() const;

    private:
        std::filesystem::path m_inputPath;

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        MPI_Comm m_comm = MPI_COMM_WORLD;
#endif
    };
} // namespace hase::openpmd
