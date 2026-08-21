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
        explicit TransportPath(std::string value = {}) : m_value(std::move(value))
        {
        }

        [[nodiscard]] TransportPath child(std::string_view name) const
        {
            return TransportPath{m_value.empty() ? std::string{name} : m_value + "/" + std::string{name}};
        }

        [[nodiscard]] std::string const& string() const
        {
            return m_value;
        }

    private:
        std::string m_value;
    };
} // namespace hase::transport
