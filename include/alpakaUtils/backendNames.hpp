/**
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <string>
#include <vector>

namespace hase::alpakaUtils
{
    /**
     * @brief Build the stable HASE name for an available Alpaka backend.
     *
     * @param backend Backend specification containing the executor selection.
     * @param device Representative device supplying the API and device kind.
     * @return `<api>_<device-kind>_<executor>` using Alpaka-provided names.
     */
    inline std::string getNameForBackend(
        alpaka::concepts::BackendSpec auto const& backend,
        alpaka::onHost::concepts::Device auto const& device)
    {
        std::string backendName;
        backendName += alpaka::onHost::getName(alpaka::getApi(device)) + "_";
        backendName += alpaka::onHost::getName(alpaka::getDeviceKind(device)) + "_";
        backendName += alpaka::onHost::getName(alpaka::getExecutor(backend));
        return backendName;
    }

    /**
     * @brief Enumerate compiled backends that currently expose a device.
     *
     * @return Stable backend names in Alpaka enumeration order. Backends with
     * no available device are omitted.
     */
    inline std::vector<std::string> availableBackendNames()
    {
        auto backends
            = alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors);
        std::vector<std::string> names;
        alpaka::onHost::executeForEachIfHasDevice(
            [&](alpaka::concepts::BackendSpec auto const& backend) -> int
            {
                auto devSelector = alpaka::onHost::makeDeviceSelector(backend);
                auto sampleDevice = devSelector.makeDevice(0);
                names.emplace_back(getNameForBackend(backend, sampleDevice));
                return 0;
            },
            backends);
        return names;
    }
} // namespace hase::alpakaUtils
