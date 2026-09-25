/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "gobot/core/math/geometry.hpp"

namespace gobot {
struct GOBOT_EXPORT TactileSensorParameters {
    std::uint32_t image_width{320};
    std::uint32_t image_height{240};
    RealType near_plane{0.0};
    RealType far_plane{0.05};
    RealType pixel_size{7.9375e-5};
    RealType density{1000.0};
    RealType young_modulus{500000.0};
    RealType poisson_ratio{0.4};
    RealType damping{0.0};
    RealType friction_coefficient{1.0};
    std::vector<std::uint32_t> coat_vertex_indices;
    std::vector<std::uint32_t> stick_vertex_indices;
    std::vector<Vector2> marker_positions;
    std::vector<std::uint32_t> marker_tetrahedra;
    std::vector<Vector4> marker_barycentric;
    std::string rgb_model{"gobot_deterministic_v1"};
    bool Validate(std::size_t vertex_count, std::size_t tetrahedron_count,
                  std::string* error = nullptr) const;
};
} // namespace gobot
