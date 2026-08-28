#pragma once

#include <core/Runtime.hpp>

#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hase::core
{
    namespace boundaryTracing
    {
        struct Direct
        {
        };

        struct Srm
        {
        };
    } // namespace boundaryTracing

    inline constexpr boundaryTracing::Direct directBoundaryTracing{};
    inline constexpr boundaryTracing::Srm srmBoundaryTracing{};

    template<typename T>
    concept BoundaryTracingPolicy = std::same_as<std::remove_cvref_t<T>, boundaryTracing::Direct>
                                    || std::same_as<std::remove_cvref_t<T>, boundaryTracing::Srm>;

    /** @brief Dispatch a runtime public setting once to a compile-time boundary policy tag. */
    decltype(auto) dispatchBoundaryTracing(AseTraceControls const& controls, auto&& operation)
    {
        if(controls.reflectionMode == "direct")
            return std::forward<decltype(operation)>(operation)(directBoundaryTracing);
        if(controls.reflectionMode == "srm")
            return std::forward<decltype(operation)>(operation)(srmBoundaryTracing);
        throw std::invalid_argument("unsupported boundary tracing policy");
    }
} // namespace hase::core
