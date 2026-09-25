/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/tactile_sensor_parameters.hpp"
#include <algorithm>
#include <cmath>

namespace gobot {
namespace {
bool SetValidationError(std::string* error, std::string message) {
    if (error) *error = std::move(message);
    return false;
}
} // namespace
bool TactileSensorParameters::Validate(
        std::size_t vertex_count, std::size_t tetrahedron_count, std::string* error) const {
    if (image_width == 0 || image_height == 0) {
        return SetValidationError(error, "tactile image resolution must be positive");
    }
    if (!std::isfinite(near_plane) || !std::isfinite(far_plane) ||
        near_plane < 0.0 || far_plane <= near_plane) {
        return SetValidationError(
                error, "tactile near/far planes must be finite and strictly ordered");
    }
    if (!std::isfinite(pixel_size) || pixel_size <= 0.0) {
        return SetValidationError(error, "tactile pixel size must be finite and positive");
    }
    if (!std::isfinite(density) || density <= 0.0 ||
        !std::isfinite(young_modulus) || young_modulus <= 0.0 ||
        !std::isfinite(poisson_ratio) || poisson_ratio <= -1.0 ||
        poisson_ratio >= 0.5 || !std::isfinite(damping) || damping < 0.0 ||
        !std::isfinite(friction_coefficient) || friction_coefficient < 0.0) {
        return SetValidationError(error, "tactile gel material parameters are invalid");
    }
    if (rgb_model.empty()) {
        return SetValidationError(error, "tactile RGB model name must not be empty");
    }
    const auto validate_vertex_set = [&](const std::vector<std::uint32_t>& indices,
                                         const char* description) {
        std::vector<std::uint32_t> sorted = indices;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            return SetValidationError(
                    error, std::string("tactile ") + description + " contains duplicates");
        }
        if (std::any_of(sorted.begin(), sorted.end(), [&](std::uint32_t index) {
                return index >= vertex_count;
            })) {
            return SetValidationError(
                    error, std::string("tactile ") + description + " references an invalid vertex");
        }
        return true;
    };
    if (!validate_vertex_set(coat_vertex_indices, "coat vertex set") ||
        !validate_vertex_set(stick_vertex_indices, "stick vertex set")) {
        return false;
    }
    if (marker_positions.size() != marker_tetrahedra.size() ||
        marker_positions.size() != marker_barycentric.size()) {
        return SetValidationError(
                error, "tactile marker position, tetrahedron, and barycentric tables must match");
    }
    for (std::size_t index = 0; index < marker_positions.size(); ++index) {
        const Vector2& marker = marker_positions[index];
        if (!marker.allFinite() || marker.x() < 0.0 || marker.y() < 0.0 ||
            marker.x() >= static_cast<RealType>(image_width) ||
            marker.y() >= static_cast<RealType>(image_height)) {
            return SetValidationError(
                    error, "tactile marker " + std::to_string(index) +
                                   " is outside the image in pixel coordinates");
        }
        if (marker_tetrahedra[index] >= tetrahedron_count) {
            return SetValidationError(
                    error, "tactile marker references an invalid gel tetrahedron");
        }
        const Vector4& barycentric = marker_barycentric[index];
        if (!barycentric.allFinite() ||
            (barycentric.array() < -CMP_EPSILON).any() ||
            std::abs(barycentric.sum() - 1.0) > 1.0e-5) {
            return SetValidationError(
                    error, "tactile marker barycentric weights must be finite, non-negative, and sum to one");
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

} // namespace gobot
