#pragma once

#include <transport/TransportReader.hpp>

namespace hase::data
{
    /** @brief Weighted wavelength samples owned by one pump source. */
    class PumpSpectrum
    {
    public:
        struct FieldName
        {
            static constexpr char const* wavelengths = "wavelengths";
            static constexpr char const* weights = "weights";
        };

        transport::Array<double> wavelengths;
        transport::Array<double> weights;

        /**
         * @brief Read weighted pump wavelengths from one graph node.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the pump-spectrum node.
         * @return Discrete spectrum in transport units.
         */
        static PumpSpectrum fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
