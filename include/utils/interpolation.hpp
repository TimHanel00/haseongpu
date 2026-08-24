/**
 * Copyright 2013 Erik Zenker, Carlchristian Eckert, Marius Melzer
 * Copyright 2026 Tim Hanel
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once
#include <vector>

#define MAX_INTERPOLATION 1000
#define LAMBDA_START 905
#define LAMBDA_STOP 1095

namespace hase::utils
{
    /**
     * @brief Linearly resample values onto an equidistant grid over the input range.
     *
     * @param y Values at the coordinates in `x`.
     * @param x Strictly increasing source coordinates.
     * @param nInterpolations Number of output samples, including both endpoints.
     * @return Resampled values ordered from `x.front()` to `x.back()`.
     * @pre `x` and `y` are non-empty, equally sized, and `nInterpolations` is
     * at least their size.
     */
    std::vector<double> interpolateLinear(
        std::vector<double> const& y,
        std::vector<double> const& x,
        unsigned nInterpolations);

} // namespace hase::utils
