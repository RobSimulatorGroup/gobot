/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "gobot/scene/node_3d.hpp"

namespace gobot {

class GOBOT_EXPORT Sensor3D : public Node3D {
    GOBCLASS(Sensor3D, Node3D)

public:
    Sensor3D() = default;

    void SetEnabled(bool enabled);

    bool IsEnabled() const;

    void SetSensorPeriod(RealType sensor_period);

    RealType GetSensorPeriod() const;

    void SetNoiseStddev(RealType noise_stddev);

    RealType GetNoiseStddev() const;

    void SetVisualizeDebug(bool visualize_debug);

    bool ShouldVisualizeDebug() const;

private:
    bool enabled_{true};
    RealType sensor_period_{0.0};
    RealType noise_stddev_{0.0};
    bool visualize_debug_{false};
};

class GOBOT_EXPORT IMUSensor3D : public Sensor3D {
    GOBCLASS(IMUSensor3D, Sensor3D)

public:
    IMUSensor3D() = default;
};

class GOBOT_EXPORT AngularMomentumSensor3D : public Sensor3D {
    GOBCLASS(AngularMomentumSensor3D, Sensor3D)

public:
    AngularMomentumSensor3D() = default;
};

class GOBOT_EXPORT ContactSensor3D : public Sensor3D {
    GOBCLASS(ContactSensor3D, Sensor3D)

public:
    ContactSensor3D() = default;

    void SetRadius(RealType radius);

    RealType GetRadius() const;

    void SetMinThreshold(RealType min_threshold);

    RealType GetMinThreshold() const;

    void SetMaxThreshold(RealType max_threshold);

    RealType GetMaxThreshold() const;

private:
    RealType radius_{0.02};
    RealType min_threshold_{0.0};
    RealType max_threshold_{0.0};
};

class GOBOT_EXPORT HeightScanner3D : public Sensor3D {
    GOBCLASS(HeightScanner3D, Sensor3D)

public:
    HeightScanner3D() = default;

    void SetSampleOffsets(const std::vector<Vector3>& sample_offsets);

    const std::vector<Vector3>& GetSampleOffsets() const;

    void SetRayDirection(const Vector3& ray_direction);

    const Vector3& GetRayDirection() const;

    void SetRayDirectionWorldSpace(bool ray_direction_world_space);

    bool IsRayDirectionWorldSpace() const;

    void SetMaxDistance(RealType max_distance);

    RealType GetMaxDistance() const;

private:
    std::vector<Vector3> sample_offsets_{Vector3::Zero()};
    Vector3 ray_direction_{0.0, 0.0, -1.0};
    bool ray_direction_world_space_{true};
    RealType max_distance_{3.0};
};

} // namespace gobot
