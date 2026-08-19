#pragma once

#include <backend/transport/TransportReader.hpp>

#include <string>

namespace hase::backend
{
    class CrossSectionTable
    {
    public:
        struct FieldName
        {
            static constexpr char const* wavelengths = "wavelengths";
            static constexpr char const* absorption = "absorption";
            static constexpr char const* emission = "emission";
            static constexpr char const* metadata = "metadata";
        };

        transport::Array<double> wavelengths;
        transport::Array<double> absorption;
        transport::Array<double> emission;
        std::string metadata;

        static CrossSectionTable fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
