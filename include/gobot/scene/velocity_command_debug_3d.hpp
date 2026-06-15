/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/node_3d.hpp"

namespace gobot {

class GOBOT_EXPORT VelocityCommandDebug3D : public Node3D {
    GOBCLASS(VelocityCommandDebug3D, Node3D)

public:
    VelocityCommandDebug3D() = default;

    void SetEnabled(bool enabled);
    bool IsEnabled() const;

    void SetShowCommandVelocity(bool show_command_velocity);
    bool ShouldShowCommandVelocity() const;

    void SetShowMeasuredVelocity(bool show_measured_velocity);
    bool ShouldShowMeasuredVelocity() const;

    void SetShowYawRate(bool show_yaw_rate);
    bool ShouldShowYawRate() const;

    void SetArrowScale(RealType arrow_scale);
    RealType GetArrowScale() const;

    void SetZOffset(RealType z_offset);
    RealType GetZOffset() const;

    void SetCommandLinearVelocity(const Vector3& command_linear_velocity);
    const Vector3& GetCommandLinearVelocity() const;

    void SetCommandYawRate(RealType command_yaw_rate);
    RealType GetCommandYawRate() const;

    void SetMeasuredLinearVelocity(const Vector3& measured_linear_velocity);
    const Vector3& GetMeasuredLinearVelocity() const;

    void SetMeasuredYawRate(RealType measured_yaw_rate);
    RealType GetMeasuredYawRate() const;

    Vector3 GetVelocityError() const;

    void SetPolicyLoaded(bool policy_loaded);
    bool IsPolicyLoaded() const;

    void SetInputFocused(bool input_focused);
    bool IsInputFocused() const;

    void SetActionNorm(RealType action_norm);
    RealType GetActionNorm() const;

private:
    bool enabled_{true};
    bool show_command_velocity_{true};
    bool show_measured_velocity_{true};
    bool show_yaw_rate_{true};
    RealType arrow_scale_{0.55};
    RealType z_offset_{0.30};

    Vector3 command_linear_velocity_{Vector3::Zero()};
    RealType command_yaw_rate_{0.0};
    Vector3 measured_linear_velocity_{Vector3::Zero()};
    RealType measured_yaw_rate_{0.0};
    bool policy_loaded_{false};
    bool input_focused_{false};
    RealType action_norm_{0.0};
};

} // namespace gobot
