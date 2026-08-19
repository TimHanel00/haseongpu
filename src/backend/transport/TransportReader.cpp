#include <backend/transport/TransportReader.hpp>
#include <openPMD/openPMD.hpp>

#include <algorithm>
#include <functional>
#include <sstream>

namespace io = openPMD;

namespace hase::backend::transport
{
    namespace
    {
        constexpr char const* recordPrefix = "hase__";
        constexpr char const* attributePrefix = "hase__attribute__";
        constexpr char const* referencePrefix = "hase__reference__";

        std::string decodePath(std::string value)
        {
            auto replaceAll = [&value](std::string const& from, std::string const& to)
            {
                std::size_t position = 0u;
                while((position = value.find(from, position)) != std::string::npos)
                {
                    value.replace(position, from.size(), to);
                    position += to.size();
                }
            };
            replaceAll("%2F", "/");
            replaceAll("%25", "%");
            return value;
        }

        std::string unquote(std::string const& value)
        {
            if(value.size() < 2u || value.front() != '"' || value.back() != '"')
                return value;
            std::string result;
            result.reserve(value.size() - 2u);
            bool escaped = false;
            for(std::size_t i = 1u; i + 1u < value.size(); ++i)
            {
                char const current = value[i];
                if(escaped)
                {
                    result.push_back(current == 'n' ? '\n' : current);
                    escaped = false;
                }
                else if(current == '\\')
                    escaped = true;
                else
                    result.push_back(current);
            }
            return result;
        }

        std::vector<std::string> parseStringArray(std::string const& value)
        {
            std::vector<std::string> result;
            std::size_t position = 0u;
            while(position < value.size())
            {
                position = value.find('"', position);
                if(position == std::string::npos)
                    break;
                std::size_t end = position + 1u;
                bool escaped = false;
                for(; end < value.size(); ++end)
                {
                    if(!escaped && value[end] == '"')
                        break;
                    escaped = !escaped && value[end] == '\\';
                    if(value[end] != '\\')
                        escaped = false;
                }
                if(end >= value.size())
                    throw std::runtime_error("invalid transport string array: " + value);
                result.push_back(unquote(value.substr(position, end - position + 1u)));
                position = end + 1u;
            }
            return result;
        }

        std::vector<std::uint64_t> parseShape(std::string const& value)
        {
            std::vector<std::uint64_t> result;
            std::size_t position = 0u;
            while(position < value.size())
            {
                position = value.find_first_of("0123456789", position);
                if(position == std::string::npos)
                    break;
                std::size_t end = position;
                while(end < value.size() && value[end] >= '0' && value[end] <= '9')
                    ++end;
                result.push_back(std::stoull(value.substr(position, end - position)));
                position = end;
            }
            return result;
        }

        std::size_t elementCount(io::Extent const& extent)
        {
            std::size_t result = 1u;
            for(auto const size : extent)
                result *= static_cast<std::size_t>(size);
            return result;
        }

        template<typename T>
        TransportReader::NumericSource numericSource(
            io::RecordComponent component,
            std::string path,
            double unitSI,
            std::vector<std::uint64_t> shape,
            std::unordered_map<std::string, TransportReader::NumericValue>& values)
        {
            auto const extent = component.getExtent();
            if(elementCount(extent) != elementCount(shape))
                throw std::runtime_error("transport field '" + path + "' has inconsistent shape metadata");
            auto sourcePath = path;
            return {
                std::move(sourcePath),
                [component, extent, path = std::move(path), unitSI, shape = std::move(shape), &values](
                    std::vector<std::function<void()>>& pending) mutable
                {
                    auto chunk = component.loadChunk<T>();
                    pending.emplace_back(
                        [chunk, extent, path, unitSI, shape, &values]
                        {
                            TransportReader::NumericValue value;
                            value.shape = shape;
                            auto const count = elementCount(extent);
                            value.values.reserve(count);
                            for(std::size_t index = 0u; index < count; ++index)
                                value.values.push_back(static_cast<double>(chunk.get()[index]) * unitSI);
                            values.emplace(path, std::move(value));
                        });
                }};
        }
    } // namespace

    TransportReader::TransportReader(io::Series& series, io::Iteration& iteration) : m_series(&series)
    {
        if(!iteration.containsAttribute("haseTransportVersion")
           || iteration.getAttribute("haseTransportVersion").get<std::string>() != "1.0")
            throw std::runtime_error("unsupported or missing HASE transport graph version");
        m_root = iteration.getAttribute("haseRoot").get<std::string>();
        auto const nodePaths = parseStringArray(iteration.getAttribute("haseNodePaths").get<std::string>());
        auto const nodeTypes = parseStringArray(iteration.getAttribute("haseNodeTypes").get<std::string>());
        if(nodePaths.size() != nodeTypes.size())
            throw std::runtime_error("transport node path/type arrays have different lengths");
        for(std::size_t index = 0u; index < nodePaths.size(); ++index)
            m_types.emplace(nodePaths[index], nodeTypes[index]);

        for(auto const& name : iteration.attributes())
        {
            if(name.starts_with(attributePrefix))
            {
                auto path = decodePath(name.substr(std::char_traits<char>::length(attributePrefix)));
                m_text.emplace(path, iteration.getAttribute(name).get<std::string>());
            }
            else if(name.starts_with(referencePrefix))
            {
                auto path = decodePath(name.substr(std::char_traits<char>::length(referencePrefix)));
                m_references.emplace(path, parseStringArray(iteration.getAttribute(name).get<std::string>()));
            }
        }

        for(auto& [name, record] : iteration.meshes)
        {
            if(!name.starts_with(recordPrefix) || !record.contains(io::MeshRecordComponent::SCALAR))
                continue;
            auto const path = record.containsAttribute("hasePath")
                                  ? record.getAttribute("hasePath").get<std::string>()
                                  : decodePath(name.substr(std::char_traits<char>::length(recordPrefix)));
            auto component = record[io::MeshRecordComponent::SCALAR];
            double const unitSI = component.unitSI();
            auto const componentExtent = component.getExtent();
            auto const shape = record.containsAttribute("haseShape")
                                   ? parseShape(record.getAttribute("haseShape").get<std::string>())
                                   : std::vector<std::uint64_t>(componentExtent.begin(), componentExtent.end());
            switch(component.getDatatype())
            {
            case io::Datatype::DOUBLE:
                m_numericSources.push_back(numericSource<double>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::FLOAT:
                m_numericSources.push_back(numericSource<float>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::INT:
                m_numericSources.push_back(numericSource<int>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::UINT:
                m_numericSources.push_back(numericSource<unsigned>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::LONGLONG:
                m_numericSources.push_back(numericSource<long long>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::ULONGLONG:
                m_numericSources.push_back(
                    numericSource<unsigned long long>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::UCHAR:
                m_numericSources.push_back(numericSource<unsigned char>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::CHAR:
                m_numericSources.push_back(numericSource<char>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::SCHAR:
                m_numericSources.push_back(numericSource<signed char>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::SHORT:
                m_numericSources.push_back(numericSource<short>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::USHORT:
                m_numericSources.push_back(numericSource<unsigned short>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::LONG:
                m_numericSources.push_back(numericSource<long>(component, path, unitSI, shape, m_numeric));
                break;
            case io::Datatype::ULONG:
                m_numericSources.push_back(numericSource<unsigned long>(component, path, unitSI, shape, m_numeric));
                break;
            default:
                throw std::runtime_error("unsupported numeric datatype for transport field '" + path + "'");
            }
        }
    }

    TransportPath TransportReader::root() const
    {
        return TransportPath{m_root};
    }

    std::string TransportReader::typeName(TransportPath const& path) const
    {
        if(auto const found = m_types.find(path.string()); found != m_types.end())
            return found->second;
        throw std::runtime_error("missing transport node type for '" + path.string() + "'");
    }

    bool TransportReader::contains(TransportPath const& prefix, char const* name) const
    {
        return containsPath(prefix.child(name).string());
    }

    bool TransportReader::containsPath(std::string const& path) const
    {
        auto const hasNumeric = [this](std::string const& candidate)
        {
            return m_numeric.contains(candidate)
                   || std::ranges::any_of(
                       m_numericSources,
                       [&candidate](auto const& source) { return source.path == candidate; });
        };
        return hasNumeric(path) || m_text.contains(path) || m_references.contains(path)
               || hasNumeric(path + "/values");
    }

    void TransportReader::prefetch(TransportPath const& prefix) const
    {
        auto const subtree = prefix.string();
        std::vector<std::function<void()>> pending;
        for(auto const& source : m_numericSources)
        {
            if(m_numeric.contains(source.path))
                continue;
            bool const belongs = source.path == subtree
                                 || (source.path.size() > subtree.size() && source.path.starts_with(subtree)
                                     && source.path[subtree.size()] == '/');
            if(belongs)
                source.schedule(pending);
        }
        if(pending.empty())
            return;
        m_series->flush();
        for(auto& complete : pending)
            complete();
    }

    TransportReader::NumericValue const& TransportReader::numeric(std::string const& path) const
    {
        if(!m_numeric.contains(path))
            prefetch(TransportPath{path});
        if(auto const found = m_numeric.find(path); found != m_numeric.end())
            return found->second;
        throw std::runtime_error("missing numeric transport field '" + path + "'");
    }

    std::string TransportReader::text(std::string const& path) const
    {
        if(auto const found = m_text.find(path); found != m_text.end())
            return unquote(found->second);
        throw std::runtime_error("missing text transport field '" + path + "'");
    }

    std::vector<std::string> TransportReader::referencePaths(std::string const& path) const
    {
        if(auto const found = m_references.find(path); found != m_references.end())
            return found->second;
        throw std::runtime_error("missing transport reference '" + path + "'");
    }

    void TransportReader::assign(std::vector<std::string>& destination, TransportPath const& prefix, char const* name)
        const
    {
        auto const path = prefix.child(name).string();
        if(auto const found = m_text.find(path); found != m_text.end())
        {
            destination = parseStringArray(found->second);
            return;
        }
        throw std::runtime_error("missing string-list transport field '" + path + "'");
    }
} // namespace hase::backend::transport
