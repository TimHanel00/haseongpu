#pragma once

#include <backend/primitives/PlanarPumpRelay.hpp>
#include <backend/primitives/Pump.hpp>
#include <backend/primitives/SurfacePumpInjector.hpp>

#include <memory>
#include <vector>

namespace hase::backend
{
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

        static PumpRegistration fromTransport(
            transport::TransportReader const& reader,
            transport::TransportPath const& prefix);
    };
} // namespace hase::backend
