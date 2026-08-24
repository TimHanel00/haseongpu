#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace hase::transport
{
    /** @brief Value type for stable paths into a primitive transport graph. */
    class TransportPath
    {
    public:
        /** @param value Slash-separated path relative to the transport root. */
        explicit TransportPath(std::string value = {}) : m_value(std::move(value))
        {
        }

        /**
         * @param name Single child field or object name.
         * @return New path with `name` appended using one slash separator.
         */
        [[nodiscard]] TransportPath child(std::string_view name) const
        {
            return TransportPath{m_value.empty() ? std::string{name} : m_value + "/" + std::string{name}};
        }

        /** @return Stored slash-separated path. */
        [[nodiscard]] std::string const& string() const
        {
            return m_value;
        }

    private:
        std::string m_value;
    };
} // namespace hase::transport
