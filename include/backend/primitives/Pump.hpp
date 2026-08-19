#pragma once

#include <backend/primitives/PumpAngularDistribution.hpp>
#include <backend/primitives/PumpProfile.hpp>
#include <backend/primitives/PumpSpectrum.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace hase::backend
{
    class Pump
    {
    public:
        struct FieldName
        {
            static constexpr char const* totalPower = "totalPower";
            static constexpr char const* rayCount = "rayCount";
            static constexpr char const* pumpSteps = "pumpSteps";
            static constexpr char const* rngSeed = "rngSeed";
            static constexpr char const* name = "name";
            static constexpr char const* spectrum = "spectrum";
            static constexpr char const* profile = "profile";
            static constexpr char const* angularDistribution = "angularDistribution";
        };

        double totalPower{};
        std::uint64_t rayCount{};
        std::optional<std::uint64_t> pumpSteps;
        std::uint64_t rngSeed{};
        std::optional<std::string> name;
        std::shared_ptr<PumpSpectrum> spectrum;
        std::shared_ptr<PumpProfile> profile;
        std::shared_ptr<PumpAngularDistribution> angularDistribution;

        static Pump fromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix);
    };
} // namespace hase::backend
