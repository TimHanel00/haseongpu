#pragma once

#include <data/VolumeTopology.hpp>
#include <transport/TransportReader.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hase::data
{
    /**
     * @brief A typed selection of entities across one or more volume topologies.
     *
     * Domain is a public graph primitive, not an execution buffer. Preparation
     * resolves its ragged masks into the cell order of one local TraceData.
     */
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

        /**
         * @brief Read one domain selection and resolve its topology references.
         * @param reader Typed reader for the active transport iteration.
         * @param prefix Path of the domain node.
         * @return Domain retaining the transported entity kind, masks, and shared topologies.
         */
        static Domain fromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix);
    };
} // namespace hase::data
