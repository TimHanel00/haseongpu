#pragma once

#include <data/Domain.hpp>

#include <memory>
#include <vector>

namespace hase::data
{
    /** @brief Optical transform connecting pump exit and entry domains. */
    class PlanarPumpRelay
    {
    public:
        struct FieldName
        {
            static constexpr char const* flipU = "flipU";
            static constexpr char const* flipV = "flipV";
            static constexpr char const* rotation = "rotation";
            static constexpr char const* offset = "offset";
            static constexpr char const* tilt = "tilt";
            static constexpr char const* magnification = "magnification";
            static constexpr char const* transmission = "transmission";
            static constexpr char const* exitDomains = "exitDomains";
            static constexpr char const* entryDomains = "entryDomains";
        };

        bool flipU{};
        bool flipV{};
        double rotation{};
        transport::Array<double> offset;
        transport::Array<double> tilt;
        double magnification{1.0};
        double transmission{1.0};
        std::vector<std::shared_ptr<Domain>> exitDomains;
        std::vector<std::shared_ptr<Domain>> entryDomains;

        /**
         * @brief Read one finite planar pump-return transform.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the relay node.
         * @return Relay parameters and identity-preserving boundary-domain references.
         */
        static PlanarPumpRelay fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
