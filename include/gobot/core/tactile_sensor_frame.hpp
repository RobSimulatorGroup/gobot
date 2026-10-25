#pragma once
#include <cstdint>
#include <vector>
#include "gobot/core/math/geometry.hpp"
namespace gobot {
// Immutable, engine-owned results. No SDK, renderer or editor objects are retained.
struct GOBOT_EXPORT TactileSensorFrame {
    std::uint64_t tick{0};
    double simulation_time{0};
    std::uint32_t width{0}, height{0};
    std::vector<std::uint8_t> rgb, rgb_marker;
    std::vector<float> depth, indentation, normal_pressure, shear_stress;
    // High-resolution collider-camera depth and TAXIM optical gel indentation.
    std::vector<float> contact_depth, optical_indentation;
    std::vector<Vector2> marker_positions, marker_motion;
    std::vector<Vector3> local_vertices;
    Vector3 total_force{Vector3::Zero()};
};

} // namespace gobot
