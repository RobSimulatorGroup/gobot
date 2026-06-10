/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/sensor_3d.hpp"

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

    Class_<Sensor3D>("Sensor3D")
            .constructor()(CtorAsRawPtr)
            .property("enabled", &Sensor3D::IsEnabled, &Sensor3D::SetEnabled)
            .property("sensor_period", &Sensor3D::GetSensorPeriod, &Sensor3D::SetSensorPeriod)
            .property("noise_stddev", &Sensor3D::GetNoiseStddev, &Sensor3D::SetNoiseStddev)
            .property("visualize_debug", &Sensor3D::ShouldVisualizeDebug, &Sensor3D::SetVisualizeDebug);

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
                      &RayCastSensor3D::SetMaxDistance);

    Class_<TerrainHeightSensor3D>("TerrainHeightSensor3D")
            .constructor()(CtorAsRawPtr)
            .property("sample_offsets", &TerrainHeightSensor3D::GetSampleOffsets,
                      &TerrainHeightSensor3D::SetSampleOffsets)
            .property("ray_direction", &TerrainHeightSensor3D::GetRayDirection,
                      &TerrainHeightSensor3D::SetRayDirection)
            .property("ray_direction_world_space", &TerrainHeightSensor3D::IsRayDirectionWorldSpace,
                      &TerrainHeightSensor3D::SetRayDirectionWorldSpace)
            .property("max_distance", &TerrainHeightSensor3D::GetMaxDistance,
                      &TerrainHeightSensor3D::SetMaxDistance)
            .property("reduction_mode", &TerrainHeightSensor3D::GetReductionMode,
                      &TerrainHeightSensor3D::SetReductionMode);

    Class_<HeightScanner3D>("HeightScanner3D")
            .constructor()(CtorAsRawPtr)
            .property("sample_offsets", &HeightScanner3D::GetSampleOffsets,
                      &HeightScanner3D::SetSampleOffsets)
            .property("ray_direction", &HeightScanner3D::GetRayDirection,
                      &HeightScanner3D::SetRayDirection)
            .property("ray_direction_world_space", &HeightScanner3D::IsRayDirectionWorldSpace,
                      &HeightScanner3D::SetRayDirectionWorldSpace)
            .property("max_distance", &HeightScanner3D::GetMaxDistance,
                      &HeightScanner3D::SetMaxDistance)
            .property("reduction_mode", &HeightScanner3D::GetReductionMode,
                      &HeightScanner3D::SetReductionMode);

};
