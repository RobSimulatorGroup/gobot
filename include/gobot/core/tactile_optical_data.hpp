#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "gobot/core/math/geometry.hpp"
namespace gobot {
struct GOBOT_EXPORT TactileOpticalData {
    std::uint32_t width{320}, height{240}, bins{0};
    std::uint32_t calibration_width{640}, calibration_height{480};
    RealType pixel_size{2.95e-5}; // metres per calibration pixel
    // RGB, magnitude bin, direction bin, six polynomial coefficients.
    std::vector<RealType> coefficients;
    // RGB interleaved linear values in [0,1], at output resolution.
    std::vector<RealType> background;
    bool Validate(std::string *error = nullptr) const;
};
} // namespace gobot
