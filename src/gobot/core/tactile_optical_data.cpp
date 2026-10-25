#include "gobot/core/tactile_optical_data.hpp"
#include <algorithm>
#include <cmath>
namespace gobot {
bool TactileOpticalData::Validate(std::string *error) const {
    auto fail = [&](const char *text) {
        if (error)
            *error = text;
        return false;
    };
    if (!width || !height || width > 4096 || height > 4096 || bins < 2 || bins > 1024 || !calibration_width ||
        !calibration_height || !std::isfinite(pixel_size) || pixel_size <= 0)
        return fail("invalid tactile optical calibration dimensions");
    if (coefficients.size() != std::size_t(3) * bins * bins * 6 ||
        background.size() != std::size_t(width) * height * 3)
        return fail("tactile optical calibration arrays do not match their dimensions");
    if (!std::all_of(coefficients.begin(), coefficients.end(), [](auto v) { return std::isfinite(v); }) ||
        !std::all_of(background.begin(), background.end(),
                     [](auto v) { return std::isfinite(v) && v >= 0 && v <= 1; }))
        return fail("tactile optical calibration contains invalid values");
    if (error)
        error->clear();
    return true;
}
} // namespace gobot
