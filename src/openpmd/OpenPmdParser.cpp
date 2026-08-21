#include <openPMD/openPMD.hpp>
#include <openpmd/OpenPmdParser.hpp>
#include <transport/TransportReader.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace io = openPMD;

namespace hase::internal::openpmd
{
    using hase::openpmd::InputSession;
    using hase::openpmd::TransportIteration;

    constexpr char const* sstConfig = R"(
{
  "backend": "adios2",
  "adios2": {
    "engine": {
      "type": "sst",
      "parameters": {
        "DataTransport": "WAN",
        "OpenTimeoutSecs": "600"
      }
    }
  }
})";

    bool hasSuffix(std::string_view value, std::string_view suffix)
    {
        return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
    }

    char const* seriesConfig(std::string const& stream)
    {
        if(hasSuffix(stream, ".sst"))
            return sstConfig;
        if(hasSuffix(stream, ".h5"))
            return R"({"backend":"hdf5"})";
        return "{}";
    }

    InputSession makeSession(std::shared_ptr<io::Series> series)
    {
        auto iterations = std::make_shared<decltype(series->readIterations())>(series->readIterations());
        auto iterator = std::make_shared<decltype(iterations->begin())>(iterations->begin());
        auto first = std::make_shared<bool>(true);
        auto closed = std::make_shared<bool>(false);
        auto simulation = std::make_shared<std::optional<data::Simulation>>();

        return InputSession{
            [series, iterations, iterator, first, closed, simulation]() mutable -> std::optional<TransportIteration>
            {
                if(*closed)
                    throw std::runtime_error("openPMD input session is closed");
                if(!*first)
                    ++*iterator;
                *first = false;
                if(*iterator == iterations->end())
                    return std::nullopt;
                auto iteration = **iterator;
                auto const index = iteration.iterationIndex;
                transport::TransportReader reader(*series, iteration);
                if(reader.dynamicOnly())
                {
                    if(!*simulation)
                        throw std::runtime_error("dynamic transport iteration arrived before the full graph");
                    (*simulation)->updateFromTransport(reader, reader.root());
                }
                else
                    *simulation = data::Simulation::fromTransport(reader, reader.root());
                iteration.close();
                return TransportIteration{index, **simulation};
            },
            [series, closed]
            {
                if(!std::exchange(*closed, true))
                    series->close();
            }};
    }
} // namespace hase::internal::openpmd

namespace hase::openpmd
{
    using namespace hase::internal::openpmd;

    InputSession::InputSession(std::function<std::optional<TransportIteration>()> next, std::function<void()> close)
        : m_next(std::move(next))
        , m_close(std::move(close))
    {
    }

    InputSession::~InputSession()
    {
        close();
    }

    std::optional<TransportIteration> InputSession::next()
    {
        if(!m_next)
            throw std::runtime_error("openPMD input session is not initialized");
        return m_next();
    }

    void InputSession::close()
    {
        if(m_close)
        {
            m_close();
            m_close = {};
            m_next = {};
        }
    }

    Parser::Parser(std::filesystem::path inputPath) : m_inputPath(std::move(inputPath))
    {
    }

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
    Parser::Parser(std::filesystem::path inputPath, MPI_Comm comm) : m_inputPath(std::move(inputPath)), m_comm(comm)
    {
    }
#endif

    InputSession Parser::open() const
    {
        auto const stream = m_inputPath.string();
#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        auto series = std::make_shared<io::Series>(stream, io::Access::READ_LINEAR, m_comm, seriesConfig(stream));
#else
        auto series = std::make_shared<io::Series>(stream, io::Access::READ_LINEAR, seriesConfig(stream));
#endif
        return makeSession(std::move(series));
    }

    data::Simulation Parser::read() const
    {
        auto session = open();
        auto iteration = session.next();
        if(!iteration)
            throw std::runtime_error("No iteration was available in the openPMD input stream.");
        auto result = std::move(iteration->simulation);
        session.close();
        return result;
    }
} // namespace hase::openpmd
