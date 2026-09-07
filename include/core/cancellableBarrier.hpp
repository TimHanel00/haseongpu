#pragma once

#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace hase::core::detail
{
    /** A reusable collective barrier whose first failure releases every participant. */
    class CancellableBarrier
    {
    public:
        explicit CancellableBarrier(unsigned const participants) : m_participants(participants)
        {
            if(participants == 0u)
                throw std::invalid_argument("a collective barrier cannot be empty");
        }

        void cancel(std::exception_ptr failure)
        {
            std::lock_guard lock(m_mutex);
            if(!m_failure)
                m_failure = std::move(failure);
            m_changed.notify_all();
        }

        void arriveAndWait()
        {
            std::unique_lock lock(m_mutex);
            auto const generation = m_generation;
            if(!m_failure && ++m_arrived == m_participants)
            {
                m_arrived = 0u;
                ++m_generation;
                m_changed.notify_all();
            }
            else
                m_changed.wait(lock, [&] { return m_failure || generation != m_generation; });
            if(m_failure)
                std::rethrow_exception(m_failure);
        }

    private:
        unsigned const m_participants;
        unsigned m_arrived = 0u;
        unsigned long long m_generation = 0u;
        std::mutex m_mutex;
        std::condition_variable m_changed;
        std::exception_ptr m_failure;
    };
} // namespace hase::core::detail
