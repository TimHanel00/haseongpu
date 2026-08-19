#pragma once

#include <backend/transport/TransportPath.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace openPMD
{
    class Iteration;
    class Series;
} // namespace openPMD

namespace hase::backend::transport
{
    template<typename T>
    struct Array
    {
        std::vector<T> values;
        std::vector<std::uint64_t> shape;
    };

    template<typename T>
    struct RaggedArray
    {
        std::vector<T> values;
        std::vector<std::uint64_t> offsets;
    };

    class TransportReader
    {
    public:
        struct NumericValue
        {
            std::vector<double> values;
            std::vector<std::uint64_t> shape;
        };

        struct NumericSource
        {
            std::string path;
            std::function<void(std::vector<std::function<void()>>&)> schedule;
        };

        TransportReader(openPMD::Series& series, openPMD::Iteration& iteration);
        TransportReader(TransportReader const&) = delete;
        TransportReader& operator=(TransportReader const&) = delete;
        TransportReader(TransportReader&&) = delete;
        TransportReader& operator=(TransportReader&&) = delete;

        [[nodiscard]] TransportPath root() const;
        [[nodiscard]] std::string typeName(TransportPath const& path) const;
        [[nodiscard]] bool contains(TransportPath const& prefix, char const* name) const;
        void prefetch(TransportPath const& prefix) const;

        template<typename T>
        void assign(T& destination, TransportPath const& prefix, char const* name) const
        {
            assignScalar(destination, prefix.child(name).string());
        }

        template<typename T>
        void assign(std::optional<T>& destination, TransportPath const& prefix, char const* name) const
        {
            auto const path = prefix.child(name).string();
            if(!containsPath(path))
            {
                destination.reset();
                return;
            }
            T value{};
            assignScalar(value, path);
            destination = std::move(value);
        }

        template<typename T>
        void assign(Array<T>& destination, TransportPath const& prefix, char const* name) const
        {
            auto const path = prefix.child(name).string();
            auto const& value = numeric(path);
            destination.values = cast<T>(value.values, path);
            destination.shape = value.shape;
        }

        template<typename T>
        void assign(std::optional<Array<T>>& destination, TransportPath const& prefix, char const* name) const
        {
            if(!contains(prefix, name))
            {
                destination.reset();
                return;
            }
            Array<T> value;
            assign(value, prefix, name);
            destination = std::move(value);
        }

        template<typename T>
        void assign(RaggedArray<T>& destination, TransportPath const& prefix, char const* name) const
        {
            auto const path = prefix.child(name).string();
            destination.values = cast<T>(numeric(path + "/values").values, path + "/values");
            destination.offsets = cast<std::uint64_t>(numeric(path + "/offsets").values, path + "/offsets");
        }

        void assign(std::vector<std::string>& destination, TransportPath const& prefix, char const* name) const;

        template<typename T>
        void assign(std::shared_ptr<T>& destination, TransportPath const& prefix, char const* name) const
        {
            auto paths = referencePaths(prefix.child(name).string());
            if(paths.empty())
            {
                destination.reset();
                return;
            }
            if(paths.size() != 1u)
            {
                throw std::runtime_error(
                    "transport reference '" + prefix.child(name).string() + "' must contain one path");
            }
            destination = object<T>(TransportPath{paths.front()});
        }

        template<typename T>
        void assign(std::vector<std::shared_ptr<T>>& destination, TransportPath const& prefix, char const* name) const
        {
            destination.clear();
            for(auto const& path : referencePaths(prefix.child(name).string()))
            {
                destination.push_back(object<T>(TransportPath{path}));
            }
        }

        template<typename T>
        [[nodiscard]] std::shared_ptr<T> object(TransportPath const& path) const
        {
            auto const key = std::string{typeid(T).name()} + "\n" + path.string();
            if(auto const found = m_objects.find(key); found != m_objects.end())
            {
                return std::static_pointer_cast<T>(found->second);
            }
            auto result = std::make_shared<T>(T::fromTransport(*this, path));
            m_objects.emplace(key, result);
            return result;
        }

        [[nodiscard]] std::vector<std::string> referencePaths(std::string const& path) const;

    private:
        [[nodiscard]] bool containsPath(std::string const& path) const;
        [[nodiscard]] NumericValue const& numeric(std::string const& path) const;
        [[nodiscard]] std::string text(std::string const& path) const;

        template<typename T>
        static std::vector<T> cast(std::vector<double> const& values, std::string const& path)
        {
            std::vector<T> result;
            result.reserve(values.size());
            for(double const value : values)
            {
                if constexpr(std::is_integral_v<T>)
                {
                    if(!std::isfinite(value) || std::trunc(value) != value)
                    {
                        throw std::runtime_error("transport field '" + path + "' is not integral");
                    }
                    auto const lowest = static_cast<long double>(std::numeric_limits<T>::lowest());
                    auto const highest = static_cast<long double>(std::numeric_limits<T>::max());
                    if(static_cast<long double>(value) < lowest || static_cast<long double>(value) > highest)
                    {
                        throw std::runtime_error("transport field '" + path + "' is outside its integer range");
                    }
                }
                result.push_back(static_cast<T>(value));
            }
            return result;
        }

        template<typename T>
        void assignScalar(T& destination, std::string const& path) const
        {
            if constexpr(std::is_same_v<T, std::string>)
            {
                destination = text(path);
            }
            else if constexpr(std::is_same_v<T, bool>)
            {
                auto const& values = numeric(path).values;
                if(values.size() != 1u || (values.front() != 0.0 && values.front() != 1.0))
                    throw std::runtime_error("transport field '" + path + "' is not boolean");
                destination = values.front() != 0.0;
            }
            else if constexpr(std::is_arithmetic_v<T>)
            {
                auto values = cast<T>(numeric(path).values, path);
                if(values.size() != 1u)
                    throw std::runtime_error("transport field '" + path + "' is not scalar");
                destination = values.front();
            }
            else
            {
                static_assert(!sizeof(T), "unsupported transport assignment type");
            }
        }

        std::string m_root;
        openPMD::Series* m_series;
        std::vector<NumericSource> m_numericSources;
        mutable std::unordered_map<std::string, NumericValue> m_numeric;
        std::unordered_map<std::string, std::string> m_text;
        std::unordered_map<std::string, std::vector<std::string>> m_references;
        std::unordered_map<std::string, std::string> m_types;
        mutable std::unordered_map<std::string, std::shared_ptr<void>> m_objects;
    };
} // namespace hase::backend::transport
