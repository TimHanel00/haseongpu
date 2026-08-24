#pragma once

#include <transport/TransportReader.hpp>

#include <string>
#include <vector>

namespace hase::data
{
    /**
     * @brief Material-owned spectral coefficients in transport SI units.
     *
     * The three arrays are one semantic value and are therefore replaced
     * together. A replacement may change the sample count; callers must
     * explicitly refresh resident material buffers at a synchronization
     * boundary before launching another trace.
     */
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
        transport::TransportPath transportPath;

        /** @brief Validate and atomically replace the complete spectral table. */
        void replaceSamples(
            std::vector<double> wavelengthValues,
            std::vector<double> absorptionValues,
            std::vector<double> emissionValues);

        /**
         * @brief Read a complete cross-section table from one graph node.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the cross-section-table node.
         * @return Validated material-owned spectral table in transport SI units.
         */
        static CrossSectionTable fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);

        /**
         * @brief Apply one explicit dynamic openPMD update to this table.
         * @param reader Typed reader for the update iteration.
         * @param prefix Path of the updated cross-section-table node.
         */
        void updateFromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix);
    };
} // namespace hase::data
