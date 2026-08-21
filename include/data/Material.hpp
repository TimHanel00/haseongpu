#pragma once

#include <data/CrossSectionTable.hpp>
#include <transport/TransportReader.hpp>

#include <memory>
#include <optional>
#include <string>

namespace hase::data
{
    /**
     * @brief Optical coefficients owned by one or more optical components.
     *
     * Cross sections remain single-owned through the shared table reference.
     * Preparation assigns a compact material id to each trace cell, allowing
     * device kernels to resolve coefficients locally at the ray wavelength.
     */
    class Material
    {
    public:
        struct FieldName
        {
            static constexpr char const* materialName = "materialName";
            static constexpr char const* temperature = "temperature";
            static constexpr char const* refractiveIndex = "refractiveIndex";
            static constexpr char const* fluorescenceLifetime = "fluorescenceLifetime";
            static constexpr char const* active = "active";
            static constexpr char const* bulkAttenuation = "bulkAttenuation";
            static constexpr char const* activeIonDensity = "activeIonDensity";
            static constexpr char const* name = "name";
            static constexpr char const* opticalAxis = "opticalAxis";
            static constexpr char const* metadata = "metadata";
            static constexpr char const* crossSections = "crossSections";
        };

        std::string materialName;
        std::optional<double> temperature;
        double refractiveIndex{};
        std::optional<double> fluorescenceLifetime;
        bool active{};
        std::optional<double> bulkAttenuation;
        double activeIonDensity{};
        std::optional<std::string> name;
        std::optional<transport::Array<double>> opticalAxis;
        std::string metadata;
        std::shared_ptr<CrossSectionTable> crossSections;

        static Material fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
