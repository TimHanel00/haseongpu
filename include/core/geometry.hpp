/**
 * Copyright 2013 Erik Zenker, Carlchristian Eckert, Marius Melzer
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @author Erik Zenker
 * @author Carlchristian Eckert
 * @author Marius Melzer
 *
 * @licence GPLv3
 **/

#pragma once
#include <alpaka/alpaka.hpp>

#include <alpakaUtils/memory.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hase::core
{
    /** @brief Two-dimensional Cartesian point or vector. */
    struct TwoDimPoint
    {
        double x;
        double y;
    };

    typedef TwoDimPoint TwoDimDir;

    /** @brief Three-dimensional Cartesian point or vector in transport coordinates. */
    struct Point
    {
        double x;
        double y;
        double z;

        constexpr Point operator-(Point const& other) const
        {
            return Point{x - other.x, y - other.y, z - other.z};
        }

        constexpr Point operator+(Point const& other) const
        {
            return Point{x + other.x, y + other.y, z + other.z};
        }

        constexpr Point operator*(double const factor) const
        {
            return Point{x * factor, y * factor, z * factor};
        }

        /** @return Euclidean norm of this vector. */
        [[nodiscard]] constexpr auto euclidLength() const
        {
            return alpaka::math::sqrt(x * x + y * y + z * z);
        }
    };

    using Position = Point;
    using Direction = Point;
    typedef Point Vector;

    /** @brief Non-owning structure-of-arrays view of Cartesian vectors. */
    template<alpaka::concepts::IView<double> T_ComponentView>
    struct CartesianViewSoA
    {
        T_ComponentView x;
        T_ComponentView y;
        T_ComponentView z;

        /**
         * @param index Element index shared by the three component views.
         * @return Cartesian value assembled from the x, y, and z components.
         */
        [[nodiscard]] ALPAKA_FN_HOST_ACC Point at(std::uint32_t const index) const
        {
            return {x[index], y[index], z[index]};
        }
    };

    template<alpaka::concepts::IView<double> T_ComponentView>
    using PositionViewSoA = CartesianViewSoA<T_ComponentView>;

    template<alpaka::concepts::IView<double> T_ComponentView>
    using DirectionViewSoA = CartesianViewSoA<T_ComponentView>;

    /** @brief Position and direction views for a structure-of-arrays ray set. */
    template<alpaka::concepts::IView<double> T_ComponentView>
    struct RayGeometryViewSoA
    {
        PositionViewSoA<T_ComponentView> positions;
        DirectionViewSoA<T_ComponentView> directions;
    };

    /** @brief Device-owned structure-of-arrays storage for Cartesian vectors. */
    template<alpaka::onHost::concepts::Device T_Device>
    class CartesianBufferSoA
    {
        using T_Buffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1u}));
        using T_View = ALPAKA_TYPEOF(std::declval<T_Buffer&>().getView());

    public:
        /**
         * @param device Device receiving all three component allocations.
         * @param extent Number of Cartesian values represented.
         */
        CartesianBufferSoA(T_Device& device, std::size_t const extent)
            : x(alpaka::onHost::alloc<double>(device, extent))
            , y(alpaka::onHost::alloc<double>(device, extent))
            , z(alpaka::onHost::alloc<double>(device, extent))
        {
        }

        /**
         * @brief Upload three host component arrays to the queue's device.
         * @param queue Queue selecting the destination device.
         * @param xValues X components in element order.
         * @param yValues Y components in element order.
         * @param zValues Z components in element order.
         */
        CartesianBufferSoA(
            hase::concepts::Queue auto const& queue,
            std::vector<double> xValues,
            std::vector<double> yValues,
            std::vector<double> zValues)
            : x(hase::alpakaUtils::toDevice(queue, std::move(xValues)))
            , y(hase::alpakaUtils::toDevice(queue, std::move(yValues)))
            , z(hase::alpakaUtils::toDevice(queue, std::move(zValues)))
        {
        }

        /** @return Non-owning component views of the three device buffers. */
        [[nodiscard]] CartesianViewSoA<T_View> view()
        {
            return {x.getView(), y.getView(), z.getView()};
        }

        T_Buffer x;
        T_Buffer y;
        T_Buffer z;
    };

    /** @brief Device-owned position and direction storage for a ray set. */
    template<alpaka::onHost::concepts::Device T_Device>
    using PositionBufferSoA = CartesianBufferSoA<T_Device>;

    template<alpaka::onHost::concepts::Device T_Device>
    using DirectionBufferSoA = CartesianBufferSoA<T_Device>;

    template<alpaka::onHost::concepts::Device T_Device>
    class RayGeometryBufferSoA
    {
        using T_ComponentView = ALPAKA_TYPEOF(std::declval<PositionBufferSoA<T_Device>&>().view().x);

    public:
        /**
         * @param positions Device-owned ray origins.
         * @param directions Device-owned ray directions in matching order.
         */
        RayGeometryBufferSoA(PositionBufferSoA<T_Device> positions, DirectionBufferSoA<T_Device> directions)
            : positions(std::move(positions))
            , directions(std::move(directions))
        {
        }

        /** @return Non-owning position and direction views for kernel arguments. */
        [[nodiscard]] RayGeometryViewSoA<T_ComponentView> view()
        {
            return {positions.view(), directions.view()};
        }

        PositionBufferSoA<T_Device> positions;
        DirectionBufferSoA<T_Device> directions;
    };

    /** @brief Host diagnostic snapshot of one unbounded ray-history segment. */
    struct InfiniteRaySnapshot
    {
        Position start;
        Position end;
        Position direction;
        double accumulatedLength;
        double totalLength;
        unsigned cell;
        double gain;
    };

    /**
     * @brief a Ray, defined by a startpoint, direction and length
     */
    struct Ray
    {
        Point p;
        Vector dir;
        double length;
    };

    /** @brief Two-dimensional ray represented by an origin and direction. */
    struct NormalRay
    {
        TwoDimPoint p;
        TwoDimDir dir;
    };

    /**
     * @param startPoint Ray origin.
     * @param endPoint Second point defining its direction.
     * @return Unit direction from `startPoint` to `endPoint`.
     */
    ALPAKA_FN_HOST_ACC Vector direction(Point startPoint, Point endPoint);

    /**
     * @param startPoint First Cartesian point.
     * @param endPoint Second Cartesian point.
     * @return Euclidean distance between the points.
     */
    ALPAKA_FN_HOST_ACC double distance(Point startPoint, Point endPoint);

    /**
     * @param startPoint Ray origin.
     * @param endPoint Ray endpoint.
     * @return Ray with direction `endPoint - startPoint` and matching length.
     */
    ALPAKA_FN_HOST_ACC Ray generateRay(Point startPoint, Point endPoint);

    /**
     * @param ray Ray whose direction and length describe one segment.
     * @return Equivalent ray with a unit direction.
     */
    ALPAKA_FN_HOST_ACC Ray normalizeRay(Ray ray);

    /** @return Cartesian dot product of `a` and `b`. */
    [[nodiscard]] constexpr double dot(Point const a, Point const b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    /** @return Right-handed Cartesian cross product of `a` and `b`. */
    [[nodiscard]] constexpr Point cross(Point const a, Point const b)
    {
        return Point{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

} // namespace hase::core
