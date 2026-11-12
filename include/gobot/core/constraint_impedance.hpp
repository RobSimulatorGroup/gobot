#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include "gobot/core/math/math_defs.hpp"

namespace gobot {
// Constraint impedance curve: minimum, maximum, width, midpoint, power.
// The endpoints are dimensionless; width uses the constraint's distance units.
inline bool IsValidConstraintImpedance(const std::vector<RealType>& values) {
    return values.size() == 5 &&
           std::all_of(values.begin(), values.end(), [](RealType v) { return std::isfinite(v); }) &&
           values[0] >= 0 && values[0] <= 1 && values[1] >= values[0] && values[1] <= 1 &&
           values[2] > 0 && values[3] > 0 && values[3] < 1 && values[4] >= 1;
}
} // namespace gobot
