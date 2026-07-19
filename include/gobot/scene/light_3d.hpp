/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/scene/node_3d.hpp"

namespace gobot {

class GOBOT_EXPORT Light3D : public Node3D {
    GOBCLASS(Light3D, Node3D)
public:
    void SetEnabled(bool enabled);
    [[nodiscard]] bool IsEnabled() const;

    void SetColor(const Color& color);
    [[nodiscard]] Color GetColor() const;

    void SetIntensity(RealType intensity);
    [[nodiscard]] RealType GetIntensity() const;

private:
    bool enabled_ = true;
    Color color_{1.0f, 1.0f, 1.0f, 1.0f};
    RealType intensity_ = 1.0;
};

class GOBOT_EXPORT DirectionalLight3D : public Light3D {
    GOBCLASS(DirectionalLight3D, Light3D)
};

class GOBOT_EXPORT PointLight3D : public Light3D {
    GOBCLASS(PointLight3D, Light3D)
public:
    void SetRange(RealType range);
    [[nodiscard]] RealType GetRange() const;

private:
    RealType range_ = 10.0;
};

class GOBOT_EXPORT SpotLight3D : public PointLight3D {
    GOBCLASS(SpotLight3D, PointLight3D)
public:
    void SetInnerAngle(RealType degrees);
    [[nodiscard]] RealType GetInnerAngle() const;

    void SetOuterAngle(RealType degrees);
    [[nodiscard]] RealType GetOuterAngle() const;

private:
    RealType inner_angle_ = 25.0;
    RealType outer_angle_ = 40.0;
};

} // namespace gobot
