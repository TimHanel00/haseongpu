#pragma once

#include <backend/transport/TransportReader.hpp>

namespace hase::backend
{
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

        static PumpSpectrum fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
