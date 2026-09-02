/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <alpakaUtils/DevBundle.hpp>
#include <alpakaUtils/backendNames.hpp>
#include <alpakaUtils/memory.hpp>
#include <alpakaUtils/utils.hpp>
#include <benchmark.hpp>
#include <core/Runtime.hpp>
#include <core/SimulationControls.hpp>
#include <core/forwardPhiAseEvaluator.hpp>
#include <core/logging.hpp>
#include <data/Simulation.hpp>
#include <data/SimulationPreparation.hpp>
#include <data/SimulationSnapshot.hpp>
#include <data/TraceData.hpp>
#include <kernels/derivativeComposition.hpp>
#include <kernels/generalPump.hpp>
#include <kernels/timeIntegrationUpdateKernels.hpp>

#include <algorithm>
#ifdef HASE_ENABLE_STEP_TIMING
#    include <chrono>
#    include <cstdlib>
#    include <fstream>
#endif
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hase::core
{
    namespace detail
    {
        /**
         * @param run Prepared simulation controls to validate before allocating devices.
         * @throws std::runtime_error If time, fields, pump sources, or synchronization settings are invalid.
         */
        inline void validateRunParameters(SimulationControls const& run)
        {
            if(run.timeStep <= 0.0)
            {
                throw std::runtime_error("simulation time_step must be positive");
            }
            if(run.numberOfSteps == 0u)
            {
                throw std::runtime_error("simulation number_of_steps must be positive");
            }
            if(run.executionMode != SimulationExecutionMode::AUTONOMOUS
               && run.executionMode != SimulationExecutionMode::SYNCHRONIZED_DEBUG)
            {
                throw std::runtime_error("simulation execution_mode must be autonomous or synchronized-debug");
            }
            unsigned previousOutputStep = 0u;
            for(unsigned outputStep : run.outputSteps)
            {
                if(outputStep == 0u || outputStep > run.numberOfSteps)
                {
                    throw std::runtime_error(
                        "simulation output_steps entries must be completed-step indices in [1, number_of_steps]");
                }
                if(outputStep <= previousOutputStep)
                {
                    throw std::runtime_error("simulation output_steps must be strictly increasing and unique");
                }
                previousOutputStep = outputStep;
            }
            auto const supportedOutputFields = SimulationOutputField::all();
            if(run.outputFields.empty())
            {
                throw std::runtime_error("simulation output_fields must contain at least one field");
            }
            for(std::size_t index = 0u; index < run.outputFields.size(); ++index)
            {
                auto const& outputField = run.outputFields[index];
                if(std::ranges::find(supportedOutputFields, outputField) == supportedOutputFields.end())
                {
                    throw std::runtime_error("unsupported simulation output field '" + outputField + "'");
                }
                if(std::ranges::find(run.outputFields.begin(), run.outputFields.begin() + index, outputField)
                   != run.outputFields.begin() + index)
                {
                    throw std::runtime_error("simulation output_fields must be unique");
                }
            }
            if(run.executionMode == SimulationExecutionMode::AUTONOMOUS && !run.controlFields.empty())
                throw std::runtime_error("simulation control_fields require synchronized-debug mode");
            if(run.executionMode == SimulationExecutionMode::SYNCHRONIZED_DEBUG && !run.outputSteps.empty())
                throw std::runtime_error(
                    "synchronized-debug emits every completed step; output_steps must be omitted");
            auto const supportedControlFields = SimulationControlField::all();
            for(std::size_t index = 0u; index < run.controlFields.size(); ++index)
            {
                auto const& controlField = run.controlFields[index];
                if(std::ranges::find(supportedControlFields, controlField) == supportedControlFields.end())
                    throw std::runtime_error("unsupported simulation control field '" + controlField + "'");
                if(std::ranges::find(run.controlFields.begin(), run.controlFields.begin() + index, controlField)
                   != run.controlFields.begin() + index)
                    throw std::runtime_error("simulation control_fields must be unique");
            }
            if(run.pump.schemaVersion != 2u || run.pump.sources.empty())
                throw std::runtime_error("invalid general pump configuration");
            for(auto const& source : run.pump.sources)
                if(source.rayCount == 0u || source.totalPower <= 0.0 || source.surfaces.empty()
                   || source.wavelengths.empty() || source.wavelengths.size() != source.spectralWeights.size()
                   || source.polarAngles.empty() || source.polarAngles.size() != source.azimuthalAngles.size()
                   || source.polarAngles.size() != source.angularWeights.size())
                    throw std::runtime_error("invalid general pump source configuration");
        }
    } // namespace detail

    /** @brief Persistent device state and integration loop for one compiled backend selection. */
    template<alpaka::onHost::concepts::Device T_Device, alpaka::concepts::Executor T_Executor>
    class CompiledSimulationRunner
    {
        using T_Queue = ALPAKA_TYPEOF(std::declval<T_Device>().makeQueue(alpaka::queueKind::nonBlocking));
        using T_DoubleBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(std::declval<T_Device&>(), std::size_t{1}));
        using T_FloatBuffer = ALPAKA_TYPEOF(alpaka::onHost::alloc<float>(std::declval<T_Device&>(), std::size_t{1}));

    public:
        /**
         * @param devices Non-empty local device set; the first device owns integration state.
         * @param executor Executor used for all kernels.
         * @param experiment Mutable ASE controls retained for each enabled ASE step.
         * @param compute Mutable execution policy retained for adaptive traces.
         * @param run Validated time, pump, output, and control configuration.
         * @param hostMesh Prepared host trace backing resident allocations.
         */
        CompiledSimulationRunner(
            std::vector<T_Device> devices,
            T_Executor const& executor,
            AseTraceControls& experiment,
            ExecutionPolicy& compute,
            SimulationControls const& run,
            hase::data::TraceData& hostMesh)
            : m_forwardAseContext(std::move(devices), executor, experiment, hostMesh)
            , m_device(m_forwardAseContext.primaryDevice())
            , m_queue(m_device.makeQueue(alpaka::queueKind::nonBlocking))
            , m_devBundle(m_device, executor)
            , m_experiment(experiment)
            , m_compute(compute)
            , m_run(run)
            , m_hostMesh(hostMesh)
            , m_mesh(m_forwardAseContext.primaryMesh().view())
            , m_beta(hase::alpakaUtils::toDevice(m_queue, hostMesh.betaVolume))
            , m_betaNext(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_stage(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_phiAse(alpaka::onHost::alloc<float>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_derivative(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_dndtPump(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_dndtAse(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_k1(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_k2(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_k3(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_k4(alpaka::onHost::alloc<double>(m_device, static_cast<std::size_t>(m_mesh.numberOfCells)))
            , m_vertexPumpIntegral(
                  alpaka::onHost::alloc<double>(
                      m_device,
                      static_cast<std::size_t>(m_mesh.numberOfMaterials) * m_mesh.numberOfMeshPoints))
            , m_generalPumpSources(
                  hase::kernels::prepareGeneralPumpDeviceSources<T_Device>(m_queue, hostMesh, m_run.pump))
        {
            if(hostMesh.betaVolume.size() != hostMesh.numberOfCells)
                throw std::runtime_error("simulation beta_volume must contain exactly one value per cell");
        }

        /**
         * @brief Advance every configured time step and publish requested snapshots.
         * @param callback Consumer invoked synchronously at output boundaries.
         * @param receiveControl Optional provider of a newly prepared trace after a completed step.
         */
        void run(
            std::function<void(data::SimulationSnapshot const&)> const& callback,
            std::function<data::TraceData(unsigned)> const& receiveControl)
        {
#ifdef HASE_ENABLE_STEP_TIMING
            std::ofstream timingCsv;
            if(auto const* timingPath = std::getenv("HASE_STEP_TIMING_CSV"))
            {
                timingCsv.open(timingPath);
                timingCsv << "revision,backend,step,elapsed_seconds,pump_enabled,ase_enabled\n";
            }
            char const* revision = std::getenv("HASE_BENCHMARK_REVISION");
            char const* backend = std::getenv("HASE_BENCHMARK_BACKEND");
#endif
            for(unsigned step = 0u; step < m_run.numberOfSteps; ++step)
            {
                unsigned const simulationStep = m_run.firstSimulationStep + step;
                bool const pumpEnabled = std::ranges::any_of(
                    m_run.pump.sources,
                    [simulationStep](auto const& source) { return simulationStep < source.pumpSteps; });
                bool const aseEnabled = simulationStep < m_run.aseSteps && !(m_run.prePump && simulationStep == 0u);
#ifdef HASE_ENABLE_STEP_TIMING
                auto const started = std::chrono::steady_clock::now();
#endif
                {
                    advanceTimeStep(simulationStep, pumpEnabled, aseEnabled);
                }
#ifdef HASE_ENABLE_STEP_TIMING
                alpaka::onHost::wait(m_queue);
                if(timingCsv)
                {
                    std::chrono::duration<double> const elapsed = std::chrono::steady_clock::now() - started;
                    timingCsv << (revision ? revision : "") << ',' << (backend ? backend : "") << ',' << (step + 1u)
                              << ',' << elapsed.count() << ',' << (pumpEnabled ? 1 : 0) << ',' << (aseEnabled ? 1 : 0)
                              << '\n';
                    timingCsv.flush();
                }
#endif
                unsigned const completedStep = step + 1u;
                m_queue.enqueueHostFn(
                    [completedStep, numberOfSteps = m_run.numberOfSteps, pumpEnabled, aseEnabled]
                    {
                        dout(V_PROGRESS) << "[HASE_STEP_COMPLETE] step=" << completedStep << '/' << numberOfSteps
                                         << " pump=" << static_cast<unsigned>(pumpEnabled)
                                         << " ase=" << static_cast<unsigned>(aseEnabled) << std::endl;
                    });
                std::swap(m_beta, m_betaNext);
                bool const synchronizedDebug = m_run.executionMode == SimulationExecutionMode::SYNCHRONIZED_DEBUG;
                if(synchronizedDebug || shouldOutput(completedStep))
                {
                    callback(makeSnapshot(completedStep));
                }
                if(synchronizedDebug && completedStep < m_run.numberOfSteps)
                {
                    if(!receiveControl)
                        throw std::runtime_error("synchronized-debug requires a control receiver");
                    auto update = receiveControl(completedStep);
                    bool const materialChanged = !m_hostMesh.hasSameMaterialData(update);
                    if(materialChanged)
                    {
                        if(!includesControl(SimulationControlField::CROSS_SECTIONS))
                            throw std::runtime_error(
                                "synchronized cross-section update requires the cross_sections control field");
                        m_hostMesh.replaceMaterialData(update);
                        m_forwardAseContext.refreshMaterials(m_hostMesh);
                        m_mesh = m_forwardAseContext.primaryMesh().view();
                    }
                    if(includesControl(SimulationControlField::BETA_VOLUME))
                    {
                        if(update.betaVolume.size() != m_mesh.numberOfCells)
                            throw std::runtime_error("synchronized beta_volume control has the wrong cell count");
                        detail::copyVectorToBuffer(m_queue, update.betaVolume, m_beta);
                        // Complete the queued copy while the callback-local host vector is alive.
                        alpaka::onHost::wait(m_queue);
                    }
                }
            }
        }

    private:
        struct DerivativeBuffers
        {
            T_DoubleBuffer& betaVolume;
            T_FloatBuffer& phiAse;
            T_DoubleBuffer& dndtPump;
            T_DoubleBuffer& dndtAse;
            T_DoubleBuffer& derivative;
        };

        /** @brief Advance one physical time step through its integration stages on the device. */
        void advanceTimeStep(unsigned const simulationStep, bool const pumpEnabled, bool const aseEnabled)
        {
            auto evaluateStage = [&](alpaka::concepts::IBuffer<double> auto& beta, bool const refreshAse = true)
            {
                if(refreshAse && aseEnabled)
                {
                    // ASE owns a separate asynchronous queue. Complete the
                    // preceding integration update before that queue reads the
                    // new stage state; pump and ASE can then run independently.
                    alpaka::onHost::wait(m_queue);
                }
                if(pumpEnabled)
                {
                    hase::kernels::enqueueGeneralPump(
                        m_devBundle,
                        m_queue,
                        m_mesh,
                        m_generalPumpSources,
                        beta,
                        m_vertexPumpIntegral,
                        m_dndtPump,
                        simulationStep);
                }
                else
                {
                    alpaka::onHost::fill(
                        m_queue,
                        m_dndtPump,
                        0.0,
                        alpaka::Vec{static_cast<std::size_t>(m_mesh.numberOfCells)});
                }

                if(refreshAse)
                {
                    initializeResult(aseEnabled ? 100000.0 : 0.0, m_hostMesh.numberOfCells);
                    if(aseEnabled)
                    {
                        m_phiAseDeviceResident
                            = m_forwardAseContext.evaluate(m_experiment, m_compute, m_hostMesh, beta, m_lastAseResult)
                                  .deviceResidentPhi;
                        if(!m_phiAseDeviceResident)
                            detail::copyVectorToBuffer(m_queue, m_lastAseResult.phiAse, m_phiAse);
                    }
                    else
                    {
                        m_phiAseDeviceResident = false;
                        alpaka::onHost::fill(
                            m_queue,
                            m_phiAse,
                            0.0f,
                            alpaka::Vec{static_cast<std::size_t>(m_mesh.numberOfCells)});
                    }
                }

                auto& activePhiAse = m_phiAseDeviceResident ? m_forwardAseContext.primaryVolumePhiAse() : m_phiAse;
                DerivativeBuffers derivativeBuffers{beta, activePhiAse, m_dndtPump, m_dndtAse, m_derivative};
                hase::kernels::enqueueComposeDerivative(m_devBundle, m_queue, m_mesh, pumpEnabled, derivativeBuffers);
            };

            auto const& method = m_run.timeIntegration.method;
            if(method == TimeIntegrator::EXPLICIT_EULER)
            {
                evaluateStage(m_beta);
                enqueueAddScaled(m_beta, m_derivative, m_betaNext, m_run.timeStep);
            }
            else if(method == TimeIntegrator::HEUN)
            {
                evaluateStage(m_beta);
                alpaka::onHost::memcpy(m_queue, m_k1, m_derivative);
                enqueueAddScaled(m_beta, m_k1, m_stage, m_run.timeStep);
                evaluateStage(m_stage);
                enqueueHeun(m_beta, m_k1, m_derivative, m_betaNext);
            }
            else if(method == TimeIntegrator::MIDPOINT)
            {
                evaluateStage(m_beta);
                enqueueAddScaled(m_beta, m_derivative, m_stage, 0.5 * m_run.timeStep);
                evaluateStage(m_stage);
                enqueueAddScaled(m_beta, m_derivative, m_betaNext, m_run.timeStep);
            }
            else if(method == TimeIntegrator::RUNGE_KUTTA_4)
            {
                integrateRungeKutta4(evaluateStage);
            }
            else if(method == TimeIntegrator::FROZEN_PHI_ASE_RUNGE_KUTTA_4)
            {
                integrateFrozenPhiAseRungeKutta4(evaluateStage);
            }
            else if(method == TimeIntegrator::FROZEN_SOURCES_RUNGE_KUTTA_4)
            {
                integrateFrozenSourcesRungeKutta4(evaluateStage);
            }
            else if(method == TimeIntegrator::IMPLICIT_EULER)
            {
                integrateImplicitEuler(evaluateStage);
            }
            else if(method == TimeIntegrator::EXPONENTIAL_EULER)
            {
                evaluateStage(m_beta);
                enqueueExponentialEuler();
            }
            else
            {
                throw std::runtime_error("unsupported time integrator '" + method + "'");
            }

            enqueueClip(m_betaNext);
        }

        void integrateRungeKutta4(std::invocable<T_DoubleBuffer&, bool> auto&& evaluateStage)
        {
            evaluateStage(m_beta);
            alpaka::onHost::memcpy(m_queue, m_k1, m_derivative);
            enqueueAddScaled(m_beta, m_k1, m_stage, 0.5 * m_run.timeStep);

            evaluateStage(m_stage);
            alpaka::onHost::memcpy(m_queue, m_k2, m_derivative);
            enqueueAddScaled(m_beta, m_k2, m_stage, 0.5 * m_run.timeStep);

            evaluateStage(m_stage);
            alpaka::onHost::memcpy(m_queue, m_k3, m_derivative);
            enqueueAddScaled(m_beta, m_k3, m_stage, m_run.timeStep);

            evaluateStage(m_stage);
            alpaka::onHost::memcpy(m_queue, m_k4, m_derivative);
            enqueueRungeKutta4();
        }

        void integrateFrozenPhiAseRungeKutta4(std::invocable<T_DoubleBuffer&, bool> auto&& evaluateStage)
        {
            evaluateStage(m_beta, true);
            alpaka::onHost::memcpy(m_queue, m_k1, m_derivative);
            enqueueAddScaled(m_beta, m_k1, m_stage, 0.5 * m_run.timeStep);

            evaluateStage(m_stage, false);
            alpaka::onHost::memcpy(m_queue, m_k2, m_derivative);
            enqueueAddScaled(m_beta, m_k2, m_stage, 0.5 * m_run.timeStep);

            evaluateStage(m_stage, false);
            alpaka::onHost::memcpy(m_queue, m_k3, m_derivative);
            enqueueAddScaled(m_beta, m_k3, m_stage, m_run.timeStep);

            evaluateStage(m_stage, false);
            alpaka::onHost::memcpy(m_queue, m_k4, m_derivative);
            enqueueRungeKutta4();
        }

        void integrateFrozenSourcesRungeKutta4(std::invocable<T_DoubleBuffer&, bool> auto&& evaluateStage)
        {
            evaluateStage(m_beta, true);
            alpaka::onHost::memcpy(m_queue, m_k1, m_derivative);
            // The source rates stay frozen, but -beta/tau changes at every stage;
            // retain each distinct slope for the final RK4 weighted sum.
            enqueueAddScaled(m_beta, m_k1, m_stage, 0.5 * m_run.timeStep);

            enqueueFrozenSourcesDerivative(m_stage, m_k2);
            enqueueAddScaled(m_beta, m_k2, m_stage, 0.5 * m_run.timeStep);

            enqueueFrozenSourcesDerivative(m_stage, m_k3);
            enqueueAddScaled(m_beta, m_k3, m_stage, m_run.timeStep);

            enqueueFrozenSourcesDerivative(m_stage, m_k4);
            enqueueRungeKutta4();
        }

        void integrateImplicitEuler(std::invocable<T_DoubleBuffer&, bool> auto&& evaluateStage)
        {
            alpaka::onHost::memcpy(m_queue, m_stage, m_beta);
            for(unsigned iteration = 0u; iteration < std::max(1u, m_run.timeIntegration.implicitIterations);
                ++iteration)
            {
                evaluateStage(m_stage);
                enqueueAddScaled(m_beta, m_derivative, m_betaNext, m_run.timeStep);
                alpaka::onHost::memcpy(m_queue, m_stage, m_betaNext);
            }
        }

        void initializeResult(double standardErrorValue, unsigned resultSize)
        {
            m_lastAseResult = data::PhiAseResult(
                std::vector<float>(resultSize, 0.0f),
                std::vector<double>(resultSize, standardErrorValue),
                std::vector<double>(resultSize, 0.0),
                std::vector<unsigned>(resultSize, 0u),
                std::vector<double>(resultSize, 0.0));
        }

        [[nodiscard]] bool includes(std::string const& field) const
        {
            return std::ranges::find(m_run.outputFields, field) != m_run.outputFields.end();
        }

        [[nodiscard]] bool includesControl(std::string const& field) const
        {
            return std::ranges::find(m_run.controlFields, field) != m_run.controlFields.end();
        }

        [[nodiscard]] bool shouldOutput(unsigned completedStep)
        {
            if(m_run.outputSteps.empty())
            {
                return true;
            }
            if(m_nextOutputStep >= m_run.outputSteps.size() || m_run.outputSteps[m_nextOutputStep] != completedStep)
            {
                return false;
            }
            ++m_nextOutputStep;
            return true;
        }

        static void copyStatus(data::PhiAseResult const& source, data::PhiAseResult& target)
        {
            target.srmStatus = source.srmStatus;
            target.srmPasses = source.srmPasses;
            target.srmRemainingFraction = source.srmRemainingFraction;
            target.srmMaxIterations = source.srmMaxIterations;
            target.srmDivergenceStreak = source.srmDivergenceStreak;
        }

        data::SimulationSnapshot makeSnapshot(unsigned step)
        {
            std::vector<double> betaVolume;
            std::vector<double> dndtPump;
            std::vector<double> dndtAse;
            data::PhiAseResult aseResult;
            copyStatus(m_lastAseResult, aseResult);

            if(includes(SimulationOutputField::BETA_VOLUME))
                betaVolume = detail::copyToVector(m_queue, m_beta);
            bool const includePhiAse = includes(SimulationOutputField::PHI_ASE);
            bool const includeStandardError = includes(SimulationOutputField::STANDARD_ERROR);
            bool const includeRelativeStandardError = includes(SimulationOutputField::RELATIVE_STANDARD_ERROR);
            bool const includeTotalRays = includes(SimulationOutputField::TOTAL_RAYS);
            if(m_phiAseDeviceResident)
            {
                auto deviceResult = m_forwardAseContext.downloadPrimaryResult(
                    includePhiAse,
                    includeStandardError,
                    includeRelativeStandardError,
                    includeTotalRays);
                aseResult.phiAse = std::move(deviceResult.phiAse);
                aseResult.standardError = std::move(deviceResult.standardError);
                aseResult.relativeStandardError = std::move(deviceResult.relativeStandardError);
                aseResult.totalRays = std::move(deviceResult.totalRays);
                aseResult.droppedRays = std::move(deviceResult.droppedRays);
            }
            else
            {
                if(includePhiAse)
                    aseResult.phiAse = m_lastAseResult.phiAse;
                if(includeStandardError)
                    aseResult.standardError = m_lastAseResult.standardError;
                if(includeRelativeStandardError)
                    aseResult.relativeStandardError = m_lastAseResult.relativeStandardError;
                if(includeTotalRays)
                {
                    aseResult.totalRays = m_lastAseResult.totalRays;
                    aseResult.droppedRays = m_lastAseResult.droppedRays;
                }
            }
            if(includes(SimulationOutputField::DNDT_ASE))
            {
                dndtAse = detail::copyToVector(m_queue, m_dndtAse);
                aseResult.dndtAse = dndtAse;
            }
            if(includes(SimulationOutputField::DNDT_PUMP))
                dndtPump = detail::copyToVector(m_queue, m_dndtPump);

            return data::SimulationSnapshot{
                step,
                static_cast<double>(step) * m_run.timeStep,
                std::move(betaVolume),
                std::move(aseResult),
                std::move(dndtPump),
                std::move(dndtAse),
                m_run.outputFields};
        }

        void enqueueAddScaled(
            alpaka::concepts::IBuffer<double> auto& base,
            alpaka::concepts::IBuffer<double> auto& slope,
            alpaka::concepts::IBuffer<double> auto& out,
            double scale)
        {
            alpaka::onHost::transform(
                m_queue,
                m_devBundle.executor,
                out,
                hase::kernels::AddScaled{scale},
                base,
                slope);
        }

        void enqueueHeun(
            alpaka::concepts::IBuffer<double> auto& base,
            alpaka::concepts::IBuffer<double> auto& first,
            alpaka::concepts::IBuffer<double> auto& second,
            alpaka::concepts::IBuffer<double> auto& out)
        {
            alpaka::onHost::transform(
                m_queue,
                m_devBundle.executor,
                out,
                hase::kernels::CombineHeun{m_run.timeStep},
                base,
                first,
                second);
        }

        void enqueueRungeKutta4()
        {
            alpaka::onHost::transform(
                m_queue,
                m_devBundle.executor,
                m_betaNext,
                hase::kernels::CombineRungeKutta4{m_run.timeStep},
                m_beta,
                m_k1,
                m_k2,
                m_k3,
                m_k4);
        }

        void enqueueFrozenSourcesDerivative(
            alpaka::concepts::IBuffer<double> auto& beta,
            alpaka::concepts::IBuffer<double> auto& stageDerivativeOut)
        {
            alpaka::onHost::transform(
                m_queue,
                m_devBundle.executor,
                stageDerivativeOut,
                hase::kernels::ComposeFrozenSourcesDerivative{
                    m_mesh.materialActive,
                    m_mesh.materialFluorescenceLifetimes},
                beta,
                m_dndtPump,
                m_dndtAse,
                m_forwardAseContext.primaryMesh().cellMaterialIds.toDeviceView());
        }

        void enqueueExponentialEuler()
        {
            alpaka::onHost::transform(
                m_queue,
                m_devBundle.executor,
                m_betaNext,
                hase::kernels::ExponentialEulerUpdate{
                    m_run.timeStep,
                    m_mesh.materialActive,
                    m_mesh.materialFluorescenceLifetimes},
                m_beta,
                m_dndtPump,
                m_dndtAse,
                m_forwardAseContext.primaryMesh().cellMaterialIds.toDeviceView());
        }

        void enqueueClip(alpaka::concepts::IBuffer<double> auto& beta)
        {
            alpaka::onHost::transform(m_queue, m_devBundle.executor, beta, hase::kernels::ClipBeta{}, beta);
        }

        ForwardPhiAseContext<T_Device, T_Executor> m_forwardAseContext;
        T_Device& m_device;
        T_Queue m_queue;
        hase::alpakaUtils::DevBundle<T_Device, T_Executor> m_devBundle;
        AseTraceControls& m_experiment;
        ExecutionPolicy& m_compute;
        SimulationControls const& m_run;
        hase::data::TraceData& m_hostMesh;
        hase::data::TraceView m_mesh;

        T_DoubleBuffer m_beta;
        T_DoubleBuffer m_betaNext;
        T_DoubleBuffer m_stage;
        T_FloatBuffer m_phiAse;
        T_DoubleBuffer m_derivative;
        T_DoubleBuffer m_dndtPump;
        T_DoubleBuffer m_dndtAse;
        T_DoubleBuffer m_k1;
        T_DoubleBuffer m_k2;
        T_DoubleBuffer m_k3;
        T_DoubleBuffer m_k4;
        T_DoubleBuffer m_vertexPumpIntegral;
        std::vector<hase::kernels::GeneralPumpDeviceSource<T_Device>> m_generalPumpSources;
        data::PhiAseResult m_lastAseResult;
        std::size_t m_nextOutputStep = 0u;
        bool m_phiAseDeviceResident = false;
    };

    namespace detail
    {
        /**
         * @brief Select a backend and run an already prepared time simulation.
         * @param experiment ASE tracing controls.
         * @param compute Mutable backend and device scheduling state.
         * @param run Time integration, pump, output, and control configuration.
         * @param hostMesh Prepared host trace retained across steps.
         * @param callback Consumer invoked for each requested output snapshot.
         * @param receiveControl Optional synchronized update provider keyed by completed step.
         * @return Zero after the matching backend completes.
         */
        inline int runPreparedSimulation(
            AseTraceControls& experiment,
            ExecutionPolicy& compute,
            SimulationControls const& run,
            hase::data::TraceData& hostMesh,
            std::function<void(data::SimulationSnapshot const&)> const& callback,
            std::function<data::TraceData(unsigned)> const& receiveControl = {})
        {
            detail::validateRunParameters(run);
            auto backends
                = alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors);
            bool oneDidRun = false;
            alpaka::onHost::executeForEachIfHasDevice(
                [&](alpaka::concepts::BackendSpec auto const& backend)
                {
                    auto const exec = alpaka::getExecutor(backend);
                    auto devSelector = alpaka::onHost::makeDeviceSelector(backend);
                    if(devSelector.getDeviceCount() == 0u)
                    {
                        return 0;
                    }
                    auto sampleDevice = devSelector.makeDevice(0);
                    if(hase::alpakaUtils::getNameForBackend(backend, sampleDevice) != compute.backend)
                    {
                        return 0;
                    }
                    std::size_t const deviceCount = devSelector.getDeviceCount();
                    compute.devices.resize(deviceCount);
                    std::iota(compute.devices.begin(), compute.devices.end(), 0u);
                    compute.gpu_i = compute.devices.front();
                    if(compute.numDevices == 0u)
                        compute.numDevices = static_cast<unsigned>(deviceCount);
                    if(compute.numDevices > deviceCount)
                    {
                        dout(V_WARNING) << "Requested number of devices (" << compute.numDevices
                                        << ") exceeds the available device count (" << deviceCount
                                        << "); using all available devices." << std::endl;
                        compute.numDevices = static_cast<unsigned>(deviceCount);
                    }
                    compute.devices.resize(compute.numDevices);
                    using T_Device = ALPAKA_TYPEOF(sampleDevice);
                    std::vector<T_Device> devices;
                    devices.reserve(compute.devices.size());
                    for(unsigned deviceIndex : compute.devices)
                        devices.emplace_back(devSelector.makeDevice(deviceIndex));
                    oneDidRun = true;
#ifdef HASE_ENABLE_BENCHMARK
                    hase::benchmark::ScopedRunContext benchmarkContext{sampleDevice, exec, compute, experiment};
#endif
                    BENCH(CompiledBackendSimulation);
                    CompiledSimulationRunner runner{std::move(devices), exec, experiment, compute, run, hostMesh};
                    runner.run(callback, receiveControl);
                    return 0;
                },
                backends);

            if(!oneDidRun)
            {
                std::ostringstream message;
                message << "Backend '" << compute.backend
                        << "' did not match any available backend with an available device. Available backends:";
                for(auto const& element : hase::alpakaUtils::availableBackendNames())
                {
                    message << "\n  " << element;
                }
                throw std::runtime_error(message.str());
            }

            return 0;
        }
    } // namespace detail

    /**
     * @brief Run a time-stepped simulation directly from the primitive graph.
     *
     * The optional control callback mutates the same graph at a synchronized
     * step boundary. Preparation then detects material-table changes and
     * refreshes only material-resident buffers; topology remains resident.
     * @param simulation Primitive graph prepared initially and after synchronized updates.
     * @param callback Consumer invoked for each requested output snapshot.
     * @param receiveControl Optional callback that mutates `simulation` at control boundaries.
     * @return Zero after successful completion.
     */
    inline int runSimulation(
        data::Simulation& simulation,
        std::function<void(data::SimulationSnapshot const&)> const& callback,
        std::function<void(unsigned, data::Simulation&)> const& receiveControl = {})
    {
        auto prepared = data::prepareSimulationWithUpdates(simulation);
        return detail::runPreparedSimulation(
            prepared.state.ase,
            prepared.state.execution,
            prepared.state.controls,
            prepared.state.trace,
            callback,
            receiveControl
                ? std::function<data::TraceData(unsigned)>{[&](unsigned const completedStep)
                                                           {
                                                               receiveControl(completedStep, simulation);
                                                               return data::prepareSimulation(simulation).trace;
                                                           }}
                : std::function<data::TraceData(unsigned)>{});
    }
} // namespace hase::core
