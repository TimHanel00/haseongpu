#pragma once

#include <transport/TransportPath.hpp>

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
#include <variant>
#include <vector>

namespace openPMD
{
    class Iteration;
    class Series;
} // namespace openPMD

namespace hase::transport
{
    /** @brief Contiguous transport values plus their declared multidimensional shape. */
    template<typename T>
    struct Array
    {
        std::vector<T> values;
        std::vector<std::uint64_t> shape;
    };

    /** @brief Concatenated variable-length transport rows and their offsets. */
    template<typename T>
    struct RaggedArray
    {
        std::vector<T> values;
        std::vector<std::uint64_t> offsets;
    };

    /**
     * @brief Typed, identity-preserving reader for one openPMD graph iteration.
     *
     * Numeric reads are prefetched in batches, while object references are
     * cached by type and transport path so shared graph nodes remain shared in
     * the C++ primitive graph. No execution-specific object is constructed by
     * this layer.
     */
    class TransportReader
    {
    public:
        struct NumericValue
        {
            using Values = std::variant<
                std::vector<char>,
                std::vector<signed char>,
                std::vector<unsigned char>,
                std::vector<short>,
                std::vector<unsigned short>,
                std::vector<int>,
                std::vector<unsigned>,
                std::vector<long>,
                std::vector<unsigned long>,
                std::vector<long long>,
                std::vector<unsigned long long>,
                std::vector<float>,
                std::vector<double>>;

            Values values;
            std::vector<std::uint64_t> shape;
            double unitSI{1.0};
        };

        struct NumericSource
        {
            std::string path;
            std::function<void(std::vector<std::function<void()>>&)> schedule;
        };

        /**
         * @param series Open openPMD series owning the iteration resources.
         * @param iteration Iteration containing one transported primitive graph.
         */
        TransportReader(openPMD::Series& series, openPMD::Iteration& iteration);
        TransportReader(TransportReader const&) = delete;
        TransportReader& operator=(TransportReader const&) = delete;
        TransportReader(TransportReader&&) = delete;
        TransportReader& operator=(TransportReader&&) = delete;

        /** @return Root path declared by the transport iteration. */
        [[nodiscard]] TransportPath root() const;

        /** @return Whether this iteration contains dynamic updates only. */
        [[nodiscard]] bool dynamicOnly() const;

        /**
         * @param path Object path to inspect.
         * @return Transported primitive type name at `path`.
         */
        [[nodiscard]] std::string typeName(TransportPath const& path) const;

        /**
         * @param prefix Object path containing the field.
         * @param name Field name relative to `prefix`.
         * @return Whether the field is present in this iteration.
         */
        [[nodiscard]] bool contains(TransportPath const& prefix, char const* name) const;

        /**
         * @brief Execute queued numeric reads below one object path.
         * @param prefix Object path whose pending numeric fields should be loaded.
         */
        void prefetch(TransportPath const& prefix) const;

        /**
         * @brief Assign one required scalar field with `unitSI` conversion.
         * @param destination Scalar replaced by the transported value.
         * @param prefix Object path containing the field.
         * @param name Field name relative to `prefix`.
         */
        template<typename T>
        void assign(T& destination, TransportPath const& prefix, char const* name) const
        {
            assignScalar(destination, prefix.child(name).string());
        }

        /**
         * @brief Assign or clear one optional scalar field.
         * @param destination Optional replaced by the value or reset when absent.
         * @param prefix Object path containing the field.
         * @param name Field name relative to `prefix`.
         */
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

        /**
         * @brief Assign one required numeric array and its declared shape.
         * @param destination Array replaced after checked conversion to `T`.
         * @param prefix Object path containing the field.
         * @param name Field name relative to `prefix`.
         */
        template<typename T>
        void assign(Array<T>& destination, TransportPath const& prefix, char const* name) const
        {
            auto const path = prefix.child(name).string();
            auto const& value = numeric(path);
            destination.values = cast<T>(value, path);
            destination.shape = value.shape;
        }

        /**
         * @brief Assign or clear one optional numeric array.
         * @param destination Optional array replaced or reset when absent.
         * @param prefix Object path containing the field.
         * @param name Field name relative to `prefix`.
         */
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

        /**
         * @brief Assign a required ragged numeric field.
         * @param destination Concatenated values and offsets to replace.
         * @param prefix Object path containing the field.
         * @param name Field name relative to `prefix`.
         */
        template<typename T>
        void assign(RaggedArray<T>& destination, TransportPath const& prefix, char const* name) const
        {
            auto const path = prefix.child(name).string();
            destination.values = cast<T>(numeric(path + "/values"), path + "/values");
            destination.offsets = cast<std::uint64_t>(numeric(path + "/offsets"), path + "/offsets");
        }

        /**
         * @brief Assign one required string array.
         * @param destination String vector to replace.
         * @param prefix Object path containing the field.
         * @param name Field name relative to `prefix`.
         */
        void assign(std::vector<std::string>& destination, TransportPath const& prefix, char const* name) const;

        /**
         * @brief Resolve one optional object reference through the identity cache.
         * @param destination Shared pointer to replace or reset when no path is stored.
         * @param prefix Object path containing the reference.
         * @param name Reference field name relative to `prefix`.
         */
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

        /**
         * @brief Resolve an ordered reference array through the identity cache.
         * @param destination Shared-pointer vector to replace.
         * @param prefix Object path containing the references.
         * @param name Reference field name relative to `prefix`.
         */
        template<typename T>
        void assign(std::vector<std::shared_ptr<T>>& destination, TransportPath const& prefix, char const* name) const
        {
            destination.clear();
            for(auto const& path : referencePaths(prefix.child(name).string()))
            {
                destination.push_back(object<T>(TransportPath{path}));
            }
        }

        /**
         * @param path Referenced primitive node.
         * @return Cached shared object, constructing it through `T::fromTransport`
         * on first access.
         */
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

        /**
         * @param path Complete path of a transported reference field.
         * @return Referenced object paths in transported order.
         */
        [[nodiscard]] std::vector<std::string> referencePaths(std::string const& path) const;

    private:
        [[nodiscard]] bool containsPath(std::string const& path) const;
        [[nodiscard]] NumericValue const& numeric(std::string const& path) const;
        [[nodiscard]] std::string text(std::string const& path) const;

        template<typename T>
        static std::vector<T> cast(NumericValue const& numericValue, std::string const& path)
        {
            std::vector<T> result;
            std::visit(
                [&](auto const& values)
                {
                    result.reserve(values.size());
                    for(auto const value : values)
                    {
                        if constexpr(std::is_integral_v<T>)
                        {
                            if constexpr(std::is_integral_v<std::remove_cvref_t<decltype(value)>>)
                            {
                                if(numericValue.unitSI == 1.0)
                                {
                                    bool inRange;
                                    if constexpr(std::is_same_v<std::remove_cvref_t<decltype(value)>, char>)
                                    {
                                        using NormalizedChar
                                            = std::conditional_t<std::is_signed_v<char>, signed char, unsigned char>;
                                        inRange = std::in_range<T>(static_cast<NormalizedChar>(value));
                                    }
                                    else
                                        inRange = std::in_range<T>(value);
                                    if(!inRange)
                                        throw std::runtime_error(
                                            "transport field '" + path + "' is outside its integer range");
                                    result.push_back(static_cast<T>(value));
                                    continue;
                                }
                            }
                            auto const scaled = static_cast<double>(value) * numericValue.unitSI;
                            if(!std::isfinite(scaled) || std::trunc(scaled) != scaled)
                                throw std::runtime_error("transport field '" + path + "' is not integral");
                            auto const lowerBound
                                = std::is_signed_v<T> ? -std::ldexp(1.0, std::numeric_limits<T>::digits) : 0.0;
                            auto const upperBound = std::ldexp(1.0, std::numeric_limits<T>::digits);
                            if(scaled < lowerBound || scaled >= upperBound)
                                throw std::runtime_error(
                                    "transport field '" + path + "' is outside its integer range");
                            result.push_back(static_cast<T>(scaled));
                        }
                        else
                            result.push_back(static_cast<T>(value) * static_cast<T>(numericValue.unitSI));
                    }
                },
                numericValue.values);
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
                auto values = cast<unsigned char>(numeric(path), path);
                if(values.size() != 1u || (values.front() != 0u && values.front() != 1u))
                    throw std::runtime_error("transport field '" + path + "' is not boolean");
                destination = values.front() != 0u;
            }
            else if constexpr(std::is_arithmetic_v<T>)
            {
                auto values = cast<T>(numeric(path), path);
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
        bool m_dynamicOnly{};
        openPMD::Series* m_series;
        std::vector<NumericSource> m_numericSources;
        mutable std::unordered_map<std::string, NumericValue> m_numeric;
        std::unordered_map<std::string, std::string> m_text;
        std::unordered_map<std::string, std::vector<std::string>> m_stringArrays;
        std::unordered_map<std::string, std::vector<std::string>> m_references;
        std::unordered_map<std::string, std::string> m_types;
        mutable std::unordered_map<std::string, std::shared_ptr<void>> m_objects;
    };
} // namespace hase::transport
