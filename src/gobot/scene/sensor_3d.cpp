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

} // namespace gobot

GOBOT_REGISTRATION {

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

};
