#pragma once

#include <data/PlanarPumpRelay.hpp>
#include <data/Pump.hpp>
#include <data/SurfacePumpInjector.hpp>

#include <memory>
#include <vector>

namespace hase::data
{
    /** @brief Associates one pump with its injection domains and optional relays. */
    class PumpRegistration
    {
    public:
        struct FieldName
        {
            static constexpr char const* pump = "pump";
            static constexpr char const* injectionMethod = "injectionMethod";
            static constexpr char const* relays = "relays";
        };

        std::shared_ptr<Pump> pump;
        std::shared_ptr<SurfacePumpInjector> injectionMethod;
        std::vector<std::shared_ptr<PlanarPumpRelay>> relays;

        /**
         * @brief Read the placement and relay configuration for one pump.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the pump-registration node.
         * @return Registration retaining shared pump, injector, and relay references.
         */
        static PumpRegistration fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::data
