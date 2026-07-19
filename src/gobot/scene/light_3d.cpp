/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/light_3d.hpp"

#include "gobot/core/registration.hpp"

#include <algorithm>

namespace gobot {

void Light3D::SetEnabled(bool enabled) { enabled_ = enabled; }
bool Light3D::IsEnabled() const { return enabled_; }

void Light3D::SetColor(const Color& color) { color_ = color; }
Color Light3D::GetColor() const { return color_; }

void Light3D::SetIntensity(RealType intensity) { intensity_ = std::max<RealType>(0.0, intensity); }
RealType Light3D::GetIntensity() const { return intensity_; }

void PointLight3D::SetRange(RealType range) { range_ = std::max<RealType>(0.001, range); }
RealType PointLight3D::GetRange() const { return range_; }

void SpotLight3D::SetInnerAngle(RealType degrees) {
    inner_angle_ = std::clamp(degrees, static_cast<RealType>(0.0), outer_angle_);
}
RealType SpotLight3D::GetInnerAngle() const { return inner_angle_; }

void SpotLight3D::SetOuterAngle(RealType degrees) {
    outer_angle_ = std::clamp(degrees, static_cast<RealType>(0.1), static_cast<RealType>(89.9));
    inner_angle_ = std::min(inner_angle_, outer_angle_);
}
RealType SpotLight3D::GetOuterAngle() const { return outer_angle_; }

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::Light3D>("Light3D")
            .constructor()(CtorAsRawPtr)
            .property("enabled", &gobot::Light3D::IsEnabled, &gobot::Light3D::SetEnabled)
            .property("color", &gobot::Light3D::GetColor, &gobot::Light3D::SetColor)
            .property("intensity", &gobot::Light3D::GetIntensity, &gobot::Light3D::SetIntensity);

    Class_<gobot::DirectionalLight3D>("DirectionalLight3D")
            .constructor()(CtorAsRawPtr);

    Class_<gobot::PointLight3D>("PointLight3D")
            .constructor()(CtorAsRawPtr)
            .property("range", &gobot::PointLight3D::GetRange, &gobot::PointLight3D::SetRange);

    Class_<gobot::SpotLight3D>("SpotLight3D")
            .constructor()(CtorAsRawPtr)
            .property("inner_angle", &gobot::SpotLight3D::GetInnerAngle, &gobot::SpotLight3D::SetInnerAngle)
            .property("outer_angle", &gobot::SpotLight3D::GetOuterAngle, &gobot::SpotLight3D::SetOuterAngle);
}
