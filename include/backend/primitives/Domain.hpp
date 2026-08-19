#pragma once

#include <backend/primitives/VolumeTopology.hpp>
#include <backend/transport/TransportReader.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hase::backend
{
    class Domain
    {
    public:
        struct FieldName
        {
            static constexpr char const* entityKind = "entityKind";
            static constexpr char const* masks = "masks";
            static constexpr char const* topologies = "topologies";
        };

        std::string entityKind;
        transport::RaggedArray<std::uint8_t> masks;
        std::vector<std::shared_ptr<VolumeTopology>> topologies;

        static Domain fromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix);
    };
} // namespace hase::backend
