#include <data/CrossSectionTable.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hase::data
{
    namespace
    {
        void validateSamples(
            std::vector<double> const& wavelengths,
            std::vector<double> const& absorption,
            std::vector<double> const& emission)
        {
            if(wavelengths.empty() || wavelengths.size() != absorption.size()
               || wavelengths.size() != emission.size())
                throw std::invalid_argument("cross-section sample arrays must have one shared non-empty extent");
            if(std::ranges::any_of(wavelengths, [](double value) { return !std::isfinite(value) || value <= 0.0; })
               || std::ranges::any_of(absorption, [](double value) { return !std::isfinite(value) || value < 0.0; })
               || std::ranges::any_of(emission, [](double value) { return !std::isfinite(value) || value < 0.0; }))
                throw std::invalid_argument("cross-section samples must be finite and physical");
            bool const monochromatic = std::ranges::all_of(
                wavelengths,
                [&](double value) { return value == wavelengths.front(); });
            if(!monochromatic)
                for(std::size_t index = 1u; index < wavelengths.size(); ++index)
                    if(wavelengths[index] <= wavelengths[index - 1u])
                        throw std::invalid_argument("cross-section wavelengths must be strictly increasing");
        }
    } // namespace

    void CrossSectionTable::replaceSamples(
        std::vector<double> wavelengthValues,
        std::vector<double> absorptionValues,
        std::vector<double> emissionValues)
    {
        validateSamples(wavelengthValues, absorptionValues, emissionValues);
        wavelengths = {std::move(wavelengthValues), {absorptionValues.size()}};
        absorption = {std::move(absorptionValues), {wavelengths.values.size()}};
        emission = {std::move(emissionValues), {wavelengths.values.size()}};
    }

    CrossSectionTable CrossSectionTable::fromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        CrossSectionTable result;
        reader.assign(result.wavelengths, prefix, FieldName::wavelengths);
        reader.assign(result.absorption, prefix, FieldName::absorption);
        reader.assign(result.emission, prefix, FieldName::emission);
        reader.assign(result.metadata, prefix, FieldName::metadata);
        result.transportPath = prefix;
        validateSamples(result.wavelengths.values, result.absorption.values, result.emission.values);
        return result;
    }

    void CrossSectionTable::updateFromTransport(
        transport::TransportReader const& reader,
        transport::TransportPath const& prefix)
    {
        if(!reader.dynamicOnly())
            throw std::runtime_error("a full transport iteration cannot update cross sections");
        reader.prefetch(prefix);
        transport::Array<double> updatedWavelengths;
        transport::Array<double> updatedAbsorption;
        transport::Array<double> updatedEmission;
        reader.assign(updatedWavelengths, prefix, FieldName::wavelengths);
        reader.assign(updatedAbsorption, prefix, FieldName::absorption);
        reader.assign(updatedEmission, prefix, FieldName::emission);
        replaceSamples(
            std::move(updatedWavelengths.values),
            std::move(updatedAbsorption.values),
            std::move(updatedEmission.values));
        transportPath = prefix;
    }
} // namespace hase::data
