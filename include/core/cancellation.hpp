/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <atomic>
#include <stdexcept>

namespace hase::core
{
    /** @brief Exception raised when cooperative cancellation reaches a polling point. */
    class OperationCancelled : public std::runtime_error
    {
    public:
        OperationCancelled() : std::runtime_error("HASEonGPU operation cancelled")
        {
        }
    };

    /** @return Process-wide cancellation flag shared by solver threads. */
    inline std::atomic_bool& cancellationRequested()
    {
        static std::atomic_bool requested{false};
        return requested;
    }

    /** @brief Request cancellation at the next cooperative polling point. */
    inline void requestCancellation()
    {
        cancellationRequested().store(true, std::memory_order_relaxed);
    }

    /** @brief Clear cancellation before beginning a new operation. */
    inline void clearCancellation()
    {
        cancellationRequested().store(false, std::memory_order_relaxed);
    }

    /** @throws OperationCancelled If cancellation is currently requested. */
    inline void throwIfCancellationRequested()
    {
        if(cancellationRequested().load(std::memory_order_relaxed))
        {
            throw OperationCancelled{};
        }
    }
} // namespace hase::core
