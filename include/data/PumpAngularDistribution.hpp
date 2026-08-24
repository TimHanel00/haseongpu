#pragma once

#include <transport/TransportReader.hpp>

namespace hase::data
{
    /** @brief Discrete weighted angular samples used when spawning pump rays. */
    class PumpAngularDistribution
    {
    public:
        struct FieldName
        {
            static constexpr char const* polarAngles = "polarAngles";
            static constexpr char const* azimuthalAngles = "azimuthalAngles";
            static constexpr char const* weights = "weights";
        };

        transport::Array<double> polarAngles;
        transport::Array<double> azimuthalAngles;
        transport::Array<double> weights;

        /**
         * @brief Read weighted polar and azimuthal launch samples.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the angular-distribution node.
         * @return Discrete angular distribution in transport units.
         */
        static PumpAngularDistribution fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
