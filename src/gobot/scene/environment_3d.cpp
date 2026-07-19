/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/environment_3d.hpp"

#include "gobot/core/registration.hpp"

#include <algorithm>

namespace gobot {

void Environment3D::SetEnabled(bool enabled) { enabled_ = enabled; }
bool Environment3D::IsEnabled() const { return enabled_; }

void Environment3D::SetClearColor(const Color& color) { clear_color_ = color; }
Color Environment3D::GetClearColor() const { return clear_color_; }

void Environment3D::SetSkyColor(const Color& color) { sky_color_ = color; }
Color Environment3D::GetSkyColor() const { return sky_color_; }

void Environment3D::SetGroundColor(const Color& color) { ground_color_ = color; }
Color Environment3D::GetGroundColor() const { return ground_color_; }

void Environment3D::SetAmbientIntensity(RealType intensity) {
    ambient_intensity_ = std::max<RealType>(0.0, intensity);
}
RealType Environment3D::GetAmbientIntensity() const { return ambient_intensity_; }

void Environment3D::SetExposure(RealType exposure) { exposure_ = std::max<RealType>(0.001, exposure); }
RealType Environment3D::GetExposure() const { return exposure_; }

void Environment3D::SetEnvironmentTexture(const Ref<Texture2D>& texture) { environment_texture_ = texture; }
const Ref<Texture2D>& Environment3D::GetEnvironmentTexture() const { return environment_texture_; }

void Environment3D::SetEnvironmentIntensity(RealType intensity) {
    environment_intensity_ = std::max<RealType>(0.0, intensity);
}
RealType Environment3D::GetEnvironmentIntensity() const { return environment_intensity_; }

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::Environment3D>("Environment3D")
            .constructor()(CtorAsRawPtr)
            .property("enabled", &gobot::Environment3D::IsEnabled, &gobot::Environment3D::SetEnabled)
            .property("clear_color", &gobot::Environment3D::GetClearColor, &gobot::Environment3D::SetClearColor)
            .property("sky_color", &gobot::Environment3D::GetSkyColor, &gobot::Environment3D::SetSkyColor)
            .property("ground_color", &gobot::Environment3D::GetGroundColor, &gobot::Environment3D::SetGroundColor)
            .property("ambient_intensity", &gobot::Environment3D::GetAmbientIntensity, &gobot::Environment3D::SetAmbientIntensity)
            .property("exposure", &gobot::Environment3D::GetExposure, &gobot::Environment3D::SetExposure)
            .property("environment_texture", &gobot::Environment3D::GetEnvironmentTexture, &gobot::Environment3D::SetEnvironmentTexture)
            .property("environment_intensity", &gobot::Environment3D::GetEnvironmentIntensity, &gobot::Environment3D::SetEnvironmentIntensity);
}
