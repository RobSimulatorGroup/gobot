/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/sensor_3d.hpp"

#include <algorithm>

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

void Sensor3D::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

bool Sensor3D::IsEnabled() const {
    return enabled_;
}

void Sensor3D::SetSensorPeriod(RealType sensor_period) {
    if (sensor_period < 0.0) {
        LOG_ERROR("Sensor3D sensor_period cannot be negative.");
        return;
    }
    sensor_period_ = sensor_period;
}

RealType Sensor3D::GetSensorPeriod() const {
    return sensor_period_;
}

void Sensor3D::SetNoiseStddev(RealType noise_stddev) {
    if (noise_stddev < 0.0) {
        LOG_ERROR("Sensor3D noise_stddev cannot be negative.");
        return;
    }
    noise_stddev_ = noise_stddev;
}

RealType Sensor3D::GetNoiseStddev() const {
    return noise_stddev_;
}

void Sensor3D::SetVisualizeDebug(bool visualize_debug) {
    visualize_debug_ = visualize_debug;
}

bool Sensor3D::ShouldVisualizeDebug() const {
    return visualize_debug_;
}

void Sensor3D::SetDebugMarkerRadius(RealType debug_marker_radius) {
    if (debug_marker_radius < 0.0) {
        LOG_ERROR("Sensor3D debug_marker_radius cannot be negative.");
        return;
    }
    debug_marker_radius_ = debug_marker_radius;
}

RealType Sensor3D::GetDebugMarkerRadius() const {
    return debug_marker_radius_;
}

void ContactSensor3D::SetRadius(RealType radius) {
    if (radius < 0.0) {
        LOG_ERROR("ContactSensor3D radius cannot be negative.");
        return;
    }
    radius_ = radius;
}

RealType ContactSensor3D::GetRadius() const {
    return radius_;
}

void ContactSensor3D::SetMinThreshold(RealType min_threshold) {
    min_threshold_ = min_threshold;
}

RealType ContactSensor3D::GetMinThreshold() const {
    return min_threshold_;
}

void ContactSensor3D::SetMaxThreshold(RealType max_threshold) {
    max_threshold_ = max_threshold;
}

RealType ContactSensor3D::GetMaxThreshold() const {
    return max_threshold_;
}

void RayCastSensor3D::SetSampleOffsets(const std::vector<Vector3>& sample_offsets) {
    sample_offsets_ = sample_offsets;
}

const std::vector<Vector3>& RayCastSensor3D::GetSampleOffsets() const {
    return sample_offsets_;
}

void RayCastSensor3D::SetRayDirection(const Vector3& ray_direction) {
    if (ray_direction.squaredNorm() <= CMP_EPSILON2) {
        LOG_ERROR("RayCastSensor3D ray_direction cannot be zero.");
        return;
    }
    ray_direction_ = ray_direction.normalized();
}

const Vector3& RayCastSensor3D::GetRayDirection() const {
    return ray_direction_;
}

void RayCastSensor3D::SetRayDirectionWorldSpace(bool ray_direction_world_space) {
    ray_direction_world_space_ = ray_direction_world_space;
}

bool RayCastSensor3D::IsRayDirectionWorldSpace() const {
    return ray_direction_world_space_;
}

void RayCastSensor3D::SetMaxDistance(RealType max_distance) {
    if (max_distance <= 0.0) {
        LOG_ERROR("RayCastSensor3D max_distance must be positive.");
        return;
    }
    max_distance_ = max_distance;
}

RealType RayCastSensor3D::GetMaxDistance() const {
    return max_distance_;
}

void RayCastSensor3D::SetPatternMode(RayPatternMode pattern_mode) {
    pattern_mode_ = pattern_mode;
}

RayPatternMode RayCastSensor3D::GetPatternMode() const {
    return pattern_mode_;
}

void RayCastSensor3D::SetGridSize(const Vector2& grid_size) {
    if (grid_size.x() <= 0.0 || grid_size.y() <= 0.0) {
        LOG_ERROR("RayCastSensor3D grid_size components must be positive.");
        return;
    }
    grid_size_ = grid_size;
}

const Vector2& RayCastSensor3D::GetGridSize() const {
    return grid_size_;
}

void RayCastSensor3D::SetGridResolution(RealType grid_resolution) {
    if (grid_resolution <= 0.0) {
        LOG_ERROR("RayCastSensor3D grid_resolution must be positive.");
        return;
    }
    grid_resolution_ = grid_resolution;
}

RealType RayCastSensor3D::GetGridResolution() const {
    return grid_resolution_;
}

void RayCastSensor3D::SetRayAlignment(RayAlignmentMode ray_alignment) {
    ray_alignment_ = ray_alignment;
}

RayAlignmentMode RayCastSensor3D::GetRayAlignment() const {
    return ray_alignment_;
}

std::vector<Vector3> RayCastSensor3D::GetResolvedSampleOffsets() const {
    if (pattern_mode_ != RayPatternMode::Grid) {
        return sample_offsets_;
    }

    const int x_count = std::max(1, static_cast<int>(std::floor(grid_size_.x() / grid_resolution_ + 0.5)) + 1);
    const int y_count = std::max(1, static_cast<int>(std::floor(grid_size_.y() / grid_resolution_ + 0.5)) + 1);
    std::vector<Vector3> offsets;
    offsets.reserve(static_cast<std::size_t>(x_count * y_count));
    for (int x_index = 0; x_index < x_count; ++x_index) {
        const RealType x = x_count == 1
                ? 0.0
                : -grid_size_.x() * 0.5 +
                  grid_size_.x() * static_cast<RealType>(x_index) / static_cast<RealType>(x_count - 1);
        for (int y_index = 0; y_index < y_count; ++y_index) {
            const RealType y = y_count == 1
                    ? 0.0
                    : -grid_size_.y() * 0.5 +
                      grid_size_.y() * static_cast<RealType>(y_index) / static_cast<RealType>(y_count - 1);
            offsets.emplace_back(x, y, 0.0);
        }
    }
    return offsets;
}

void TerrainHeightSensor3D::SetReductionMode(RayReductionMode reduction_mode) {
    reduction_mode_ = reduction_mode;
}

RayReductionMode TerrainHeightSensor3D::GetReductionMode() const {
    return reduction_mode_;
}

HeightScanner3D::HeightScanner3D() {
    SetReductionMode(RayReductionMode::None);
}

} // namespace gobot

GOBOT_REGISTRATION {

    gobot::QuickEnumeration_<gobot::RayReductionMode>("RayReductionMode");
    gobot::QuickEnumeration_<gobot::RayPatternMode>("RayPatternMode");
    gobot::QuickEnumeration_<gobot::RayAlignmentMode>("RayAlignmentMode");

    Class_<Sensor3D>("Sensor3D")
            .constructor()(CtorAsRawPtr)
            .property("enabled", &Sensor3D::IsEnabled, &Sensor3D::SetEnabled)
            .property("sensor_period", &Sensor3D::GetSensorPeriod, &Sensor3D::SetSensorPeriod)
            .property("noise_stddev", &Sensor3D::GetNoiseStddev, &Sensor3D::SetNoiseStddev)
            .property("visualize_debug", &Sensor3D::ShouldVisualizeDebug, &Sensor3D::SetVisualizeDebug)
            .property("debug_marker_radius", &Sensor3D::GetDebugMarkerRadius, &Sensor3D::SetDebugMarkerRadius);

    Class_<IMUSensor3D>("IMUSensor3D")
            .constructor()(CtorAsRawPtr);

    Class_<AngularMomentumSensor3D>("AngularMomentumSensor3D")
            .constructor()(CtorAsRawPtr);

    Class_<ContactSensor3D>("ContactSensor3D")
            .constructor()(CtorAsRawPtr)
            .property("radius", &ContactSensor3D::GetRadius, &ContactSensor3D::SetRadius)
            .property("min_threshold", &ContactSensor3D::GetMinThreshold, &ContactSensor3D::SetMinThreshold)
            .property("max_threshold", &ContactSensor3D::GetMaxThreshold, &ContactSensor3D::SetMaxThreshold);

    Class_<RayCastSensor3D>("RayCastSensor3D")
            .constructor()(CtorAsRawPtr)
            .property("sample_offsets", &RayCastSensor3D::GetSampleOffsets,
                      &RayCastSensor3D::SetSampleOffsets)
            .property("ray_direction", &RayCastSensor3D::GetRayDirection,
                      &RayCastSensor3D::SetRayDirection)
            .property("ray_direction_world_space", &RayCastSensor3D::IsRayDirectionWorldSpace,
                      &RayCastSensor3D::SetRayDirectionWorldSpace)
            .property("max_distance", &RayCastSensor3D::GetMaxDistance,
                      &RayCastSensor3D::SetMaxDistance)
            .property("pattern_mode", &RayCastSensor3D::GetPatternMode,
                      &RayCastSensor3D::SetPatternMode)
            .property("grid_size", &RayCastSensor3D::GetGridSize,
                      &RayCastSensor3D::SetGridSize)
            .property("grid_resolution", &RayCastSensor3D::GetGridResolution,
                      &RayCastSensor3D::SetGridResolution)
            .property("ray_alignment", &RayCastSensor3D::GetRayAlignment,
                      &RayCastSensor3D::SetRayAlignment);

    Class_<TerrainHeightSensor3D>("TerrainHeightSensor3D")
            .constructor()(CtorAsRawPtr)
            .property("reduction_mode", &TerrainHeightSensor3D::GetReductionMode,
                      &TerrainHeightSensor3D::SetReductionMode);

    Class_<HeightScanner3D>("HeightScanner3D")
            .constructor()(CtorAsRawPtr);

};
