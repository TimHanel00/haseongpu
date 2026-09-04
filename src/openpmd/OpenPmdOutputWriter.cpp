#include <core/SimulationControls.hpp>
#include <openPMD/openPMD.hpp>
#include <openpmd/OpenPmdOutputWriter.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>

namespace io = openPMD;

namespace hase::internal::openpmd::output
{
    constexpr char const* transportVersion = "1.0";

    bool hasSuffix(std::string_view value, std::string_view suffix)
    {
        return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
    }

    std::string seriesConfig(std::string const& stream)
    {
        if(hasSuffix(stream, ".sst"))
            return R"({"backend":"adios2","adios2":{"engine":{"type":"sst","parameters":{"DataTransport":"WAN","OpenTimeoutSecs":"600","QueueLimit":"0"}}}})";
        if(hasSuffix(stream, ".h5"))
            return R"({"backend":"hdf5"})";
        return "{}";
    }

    std::string encodePath(std::string const& path)
    {
        std::string result;
        result.reserve(path.size());
        for(char character : path)
        {
            if(character == '%')
                result += "%25";
            else if(character == '/')
                result += "%2F";
            else
                result += character;
        }
        return result;
    }

    template<typename T>
    void writeArray(io::Iteration& iteration, std::string const& path, std::vector<T> const& values)
    {
        auto record = iteration.meshes["hase__" + encodePath(path)];
        record.setAttribute("haseTransportVersion", std::string{transportVersion});
        record.setAttribute("hasePath", path);
        record.setAttribute("haseFieldName", path.substr(path.rfind('/') + 1u));
        record.setAttribute("haseDynamic", true);
        record.setAttribute("haseShape", "[" + std::to_string(values.size()) + "]");
        record.setAxisLabels({"flatIndex"});
        record.setGridSpacing(std::vector<double>{1.0});
        record.setGridGlobalOffset(std::vector<double>{0.0});
        record.setGridUnitSI(1.0);
        auto& component = record[io::MeshRecordComponent::SCALAR];
        component.setUnitSI(1.0);
        component.setPosition(std::vector<double>{0.0});
        component.resetDataset(io::Dataset{io::determineDatatype<T>(), io::Extent{values.size()}});
        auto data = std::shared_ptr<T[]>(new T[values.size()]);
        std::copy(values.begin(), values.end(), data.get());
        component.storeChunk(data, io::Offset{0u}, io::Extent{values.size()});
    }

    void writeResultStatus(io::Iteration& iteration, std::string const& root, data::PhiAseResult const& result)
    {
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryStatus"),
            std::string{data::toString(result.boundaryStatus)});
        iteration.setAttribute("hase__attribute__" + encodePath(root + "/boundaryPasses"), result.boundaryPasses);
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryRemainingFraction"),
            result.boundaryRemainingFraction);
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryMaxPasses"),
            result.boundaryMaxPasses);
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryDivergenceStreak"),
            result.boundaryDivergenceStreak);
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryTailStatus"),
            std::string{data::toString(result.boundaryTailStatus)});
        iteration.setAttribute("hase__attribute__" + encodePath(root + "/boundaryGamma"), result.boundaryGamma);
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryGammaStandardError"),
            result.boundaryGammaStandardError);
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryTailFactor"),
            result.boundaryTailFactor);
        iteration.setAttribute(
            "hase__attribute__" + encodePath(root + "/boundaryTailClosure"),
            result.boundaryTailClosure);
    }

    void setRoot(io::Iteration& iteration, std::string const& root)
    {
        iteration.setAttribute("haseTransportVersion", std::string{transportVersion});
        iteration.setAttribute("haseRoot", root);
        iteration.setAttribute("haseNodePaths", "[\"" + root + "\"]");
        iteration.setAttribute("haseNodeTypes", "[\"" + root + "\"]");
    }
} // namespace hase::internal::openpmd::output

namespace hase::openpmd
{
    using namespace hase::internal::openpmd::output;

    class OutputWriter::Impl
    {
    public:
        explicit Impl(std::filesystem::path path)
        {
            auto const stream = path.string();
            series = std::make_unique<io::Series>(stream, io::Access::CREATE_LINEAR, seriesConfig(stream));
            series->setAttribute("haseTransportVersion", std::string{transportVersion});
        }

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
        Impl(std::filesystem::path path, MPI_Comm comm)
        {
            auto const stream = path.string();
            series = std::make_unique<io::Series>(stream, io::Access::CREATE_LINEAR, comm, seriesConfig(stream));
            series->setAttribute("haseTransportVersion", std::string{transportVersion});
        }
#endif

        std::unique_ptr<io::Series> series;
    };

    OutputWriter::OutputWriter(std::filesystem::path outputPath)
        : m_impl(std::make_unique<Impl>(std::move(outputPath)))
    {
    }

#if defined(MPI_FOUND) && !defined(DISABLE_MPI)
    OutputWriter::OutputWriter(std::filesystem::path outputPath, MPI_Comm comm)
        : m_impl(std::make_unique<Impl>(std::move(outputPath), comm))
    {
    }
#endif

    OutputWriter::OutputWriter(OutputWriter&&) noexcept = default;
    OutputWriter& OutputWriter::operator=(OutputWriter&&) noexcept = default;
    OutputWriter::~OutputWriter() = default;

    void OutputWriter::writeResult(std::uint64_t iterationIndex, data::PhiAseResult const& result)
    {
        auto iteration = m_impl->series->snapshots()[iterationIndex];
        setRoot(iteration, "phiAseResult");
        writeArray(iteration, "phiAseResult/phiAse", result.phiAse);
        writeArray(iteration, "phiAseResult/standardError", result.standardError);
        writeArray(iteration, "phiAseResult/relativeStandardError", result.relativeStandardError);
        writeArray(iteration, "phiAseResult/totalRays", result.totalRays);
        writeArray(iteration, "phiAseResult/dndtAse", result.dndtAse);
        writeResultStatus(iteration, "phiAseResult", result);
        iteration.close();
        m_impl->series->flush();
    }

    void OutputWriter::writeSnapshot(std::uint64_t iterationIndex, data::SimulationSnapshot const& snapshot)
    {
        auto iteration = m_impl->series->snapshots()[iterationIndex];
        setRoot(iteration, "simulationSnapshot");
        iteration.setAttribute("hase__attribute__simulationSnapshot%2Fstep", snapshot.step);
        iteration.setAttribute("hase__attribute__simulationSnapshot%2Ftime", snapshot.time);
        if(snapshot.contains(core::SimulationOutputField::BETA_VOLUME))
            writeArray(iteration, "simulationSnapshot/betaVolume", snapshot.betaVolume);
        bool const hasAseResult = snapshot.contains(core::SimulationOutputField::PHI_ASE)
                                  || snapshot.contains(core::SimulationOutputField::STANDARD_ERROR)
                                  || snapshot.contains(core::SimulationOutputField::RELATIVE_STANDARD_ERROR)
                                  || snapshot.contains(core::SimulationOutputField::TOTAL_RAYS)
                                  || snapshot.contains(core::SimulationOutputField::DNDT_ASE);
        if(snapshot.contains(core::SimulationOutputField::PHI_ASE))
            writeArray(iteration, "simulationSnapshot/phiAse", snapshot.aseResult.phiAse);
        if(snapshot.contains(core::SimulationOutputField::STANDARD_ERROR))
            writeArray(iteration, "simulationSnapshot/standardError", snapshot.aseResult.standardError);
        if(snapshot.contains(core::SimulationOutputField::RELATIVE_STANDARD_ERROR))
            writeArray(
                iteration,
                "simulationSnapshot/relativeStandardError",
                snapshot.aseResult.relativeStandardError);
        if(snapshot.contains(core::SimulationOutputField::TOTAL_RAYS))
            writeArray(iteration, "simulationSnapshot/totalRays", snapshot.aseResult.totalRays);
        if(snapshot.contains(core::SimulationOutputField::DNDT_ASE))
            writeArray(iteration, "simulationSnapshot/dndtAse", snapshot.aseResult.dndtAse);
        if(hasAseResult)
            writeResultStatus(iteration, "simulationSnapshot", snapshot.aseResult);
        if(snapshot.contains(core::SimulationOutputField::DNDT_PUMP))
            writeArray(iteration, "simulationSnapshot/dndtPump", snapshot.dndtPump);
        iteration.close();
        m_impl->series->flush();
    }

    void OutputWriter::close()
    {
        if(m_impl && m_impl->series)
        {
            m_impl->series->close();
            m_impl->series.reset();
        }
    }
} // namespace hase::openpmd
