/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

} // namespace gobot
