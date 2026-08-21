#include <data/Material.hpp>

namespace hase::data
{
    Material Material::fromTransport(transport::TransportReader const& reader, transport::TransportPath const& prefix)
    {
        reader.prefetch(prefix);
        Material result;
        reader.assign(result.materialName, prefix, FieldName::materialName);
        reader.assign(result.temperature, prefix, FieldName::temperature);
        reader.assign(result.refractiveIndex, prefix, FieldName::refractiveIndex);
        reader.assign(result.fluorescenceLifetime, prefix, FieldName::fluorescenceLifetime);
        reader.assign(result.active, prefix, FieldName::active);
        reader.assign(result.bulkAttenuation, prefix, FieldName::bulkAttenuation);
        reader.assign(result.activeIonDensity, prefix, FieldName::activeIonDensity);
        reader.assign(result.name, prefix, FieldName::name);
        reader.assign(result.opticalAxis, prefix, FieldName::opticalAxis);
        reader.assign(result.metadata, prefix, FieldName::metadata);
        reader.assign(result.crossSections, prefix, FieldName::crossSections);
        return result;
    }
} // namespace hase::data
