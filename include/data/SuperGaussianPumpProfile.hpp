#pragma once

#include <transport/TransportReader.hpp>

#include <string>

namespace hase::data
{
    /** @brief Oriented super-Gaussian spatial pump profile in transport units. */
    class SuperGaussianPumpProfile
    {
    public:
        struct FieldName
        {
            static constexpr char const* kind = "kind";
            static constexpr char const* radiusU = "radiusU";
            static constexpr char const* radiusV = "radiusV";
            static constexpr char const* exponent = "exponent";
            static constexpr char const* center = "center";
            static constexpr char const* axisU = "axisU";
            static constexpr char const* axisV = "axisV";
        };

        std::string kind;
        double radiusU{};
        double radiusV{};
        double exponent{};
        transport::Array<double> center;
        transport::Array<double> axisU;
        transport::Array<double> axisV;

        /**
         * @brief Read an oriented super-Gaussian profile from one graph node.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the profile node.
         * @return Profile parameters in transport units.
         */
        static SuperGaussianPumpProfile fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
