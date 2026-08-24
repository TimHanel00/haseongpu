#pragma once

#include <string>
#include <vector>

namespace hase::openpmd
{
    /** @return Names of openPMD backends supported by the linked provider. */
    std::vector<std::string> availableOpenPmdBackends();
} // namespace hase::openpmd
