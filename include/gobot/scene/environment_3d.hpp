/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/texture.hpp"

namespace gobot {

class GOBOT_EXPORT Environment3D : public Node3D {
    GOBCLASS(Environment3D, Node3D)
public:
    void SetEnabled(bool enabled);
    [[nodiscard]] bool IsEnabled() const;

    void SetClearColor(const Color& color);
    [[nodiscard]] Color GetClearColor() const;

    void SetSkyColor(const Color& color);
    [[nodiscard]] Color GetSkyColor() const;

    void SetGroundColor(const Color& color);
    [[nodiscard]] Color GetGroundColor() const;

    void SetAmbientIntensity(RealType intensity);
    [[nodiscard]] RealType GetAmbientIntensity() const;

    void SetExposure(RealType exposure);
    [[nodiscard]] RealType GetExposure() const;

    void SetEnvironmentTexture(const Ref<Texture2D>& texture);
    [[nodiscard]] const Ref<Texture2D>& GetEnvironmentTexture() const;

    void SetEnvironmentIntensity(RealType intensity);
    [[nodiscard]] RealType GetEnvironmentIntensity() const;

private:
    bool enabled_ = true;
    Color clear_color_{0.075f, 0.08f, 0.09f, 1.0f};
    Color sky_color_{0.58f, 0.67f, 0.78f, 1.0f};
    Color ground_color_{0.09f, 0.095f, 0.105f, 1.0f};
    RealType ambient_intensity_ = 0.38;
    RealType exposure_ = 0.9;
    Ref<Texture2D> environment_texture_;
    RealType environment_intensity_ = 1.0;
};

} // namespace gobot
