/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/velocity_command_debug_3d.hpp"

#include <algorithm>

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

void VelocityCommandDebug3D::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

bool VelocityCommandDebug3D::IsEnabled() const {
    return enabled_;
}

void VelocityCommandDebug3D::SetShowCommandVelocity(bool show_command_velocity) {
    show_command_velocity_ = show_command_velocity;
}

bool VelocityCommandDebug3D::ShouldShowCommandVelocity() const {
    return show_command_velocity_;
}

void VelocityCommandDebug3D::SetShowMeasuredVelocity(bool show_measured_velocity) {
    show_measured_velocity_ = show_measured_velocity;
}

bool VelocityCommandDebug3D::ShouldShowMeasuredVelocity() const {
    return show_measured_velocity_;
}

void VelocityCommandDebug3D::SetShowYawRate(bool show_yaw_rate) {
    show_yaw_rate_ = show_yaw_rate;
}

bool VelocityCommandDebug3D::ShouldShowYawRate() const {
    return show_yaw_rate_;
}

void VelocityCommandDebug3D::SetArrowScale(RealType arrow_scale) {
    if (arrow_scale < 0.0) {
        LOG_ERROR("VelocityCommandDebug3D arrow_scale cannot be negative.");
        return;
    }
    arrow_scale_ = arrow_scale;
}

RealType VelocityCommandDebug3D::GetArrowScale() const {
    return arrow_scale_;
}

void VelocityCommandDebug3D::SetZOffset(RealType z_offset) {
    z_offset_ = z_offset;
}

RealType VelocityCommandDebug3D::GetZOffset() const {
    return z_offset_;
}

void VelocityCommandDebug3D::SetCommandLinearVelocity(const Vector3& command_linear_velocity) {
    command_linear_velocity_ = command_linear_velocity;
}

const Vector3& VelocityCommandDebug3D::GetCommandLinearVelocity() const {
    return command_linear_velocity_;
}

void VelocityCommandDebug3D::SetCommandYawRate(RealType command_yaw_rate) {
    command_yaw_rate_ = command_yaw_rate;
}

RealType VelocityCommandDebug3D::GetCommandYawRate() const {
    return command_yaw_rate_;
}

void VelocityCommandDebug3D::SetMeasuredLinearVelocity(const Vector3& measured_linear_velocity) {
    measured_linear_velocity_ = measured_linear_velocity;
}

const Vector3& VelocityCommandDebug3D::GetMeasuredLinearVelocity() const {
    return measured_linear_velocity_;
}

void VelocityCommandDebug3D::SetMeasuredYawRate(RealType measured_yaw_rate) {
    measured_yaw_rate_ = measured_yaw_rate;
}

RealType VelocityCommandDebug3D::GetMeasuredYawRate() const {
    return measured_yaw_rate_;
}

Vector3 VelocityCommandDebug3D::GetVelocityError() const {
    return command_linear_velocity_ - measured_linear_velocity_;
}

void VelocityCommandDebug3D::SetPolicyLoaded(bool policy_loaded) {
    policy_loaded_ = policy_loaded;
}

bool VelocityCommandDebug3D::IsPolicyLoaded() const {
    return policy_loaded_;
}

void VelocityCommandDebug3D::SetInputFocused(bool input_focused) {
    input_focused_ = input_focused;
}

bool VelocityCommandDebug3D::IsInputFocused() const {
    return input_focused_;
}

void VelocityCommandDebug3D::SetActionNorm(RealType action_norm) {
    action_norm_ = std::max<RealType>(0.0, action_norm);
}

RealType VelocityCommandDebug3D::GetActionNorm() const {
    return action_norm_;
}

} // namespace gobot

GOBOT_REGISTRATION {

    USING_ENUM_BITWISE_OPERATORS;

    auto editor_only = []() {
        return gobot::AddMetaPropertyInfo(
                gobot::PropertyInfo().SetUsageFlags(gobot::PropertyUsageFlags::Editor));
    };

    Class_<gobot::VelocityCommandDebug3D>("VelocityCommandDebug3D")
            .constructor()(gobot::CtorAsRawPtr)
            .property("enabled",
                      &gobot::VelocityCommandDebug3D::IsEnabled,
                      &gobot::VelocityCommandDebug3D::SetEnabled)
            .property("show_command_velocity",
                      &gobot::VelocityCommandDebug3D::ShouldShowCommandVelocity,
                      &gobot::VelocityCommandDebug3D::SetShowCommandVelocity)
            .property("show_measured_velocity",
                      &gobot::VelocityCommandDebug3D::ShouldShowMeasuredVelocity,
                      &gobot::VelocityCommandDebug3D::SetShowMeasuredVelocity)
            .property("show_yaw_rate",
                      &gobot::VelocityCommandDebug3D::ShouldShowYawRate,
                      &gobot::VelocityCommandDebug3D::SetShowYawRate)
            .property("arrow_scale",
                      &gobot::VelocityCommandDebug3D::GetArrowScale,
                      &gobot::VelocityCommandDebug3D::SetArrowScale)
            .property("z_offset",
                      &gobot::VelocityCommandDebug3D::GetZOffset,
                      &gobot::VelocityCommandDebug3D::SetZOffset)
            .property("command_linear_velocity",
                      &gobot::VelocityCommandDebug3D::GetCommandLinearVelocity,
                      &gobot::VelocityCommandDebug3D::SetCommandLinearVelocity)(editor_only())
            .property("command_yaw_rate",
                      &gobot::VelocityCommandDebug3D::GetCommandYawRate,
                      &gobot::VelocityCommandDebug3D::SetCommandYawRate)(editor_only())
            .property("measured_linear_velocity",
                      &gobot::VelocityCommandDebug3D::GetMeasuredLinearVelocity,
                      &gobot::VelocityCommandDebug3D::SetMeasuredLinearVelocity)(editor_only())
            .property("measured_yaw_rate",
                      &gobot::VelocityCommandDebug3D::GetMeasuredYawRate,
                      &gobot::VelocityCommandDebug3D::SetMeasuredYawRate)(editor_only())
            .property_readonly("velocity_error",
                      &gobot::VelocityCommandDebug3D::GetVelocityError)(editor_only())
            .property("policy_loaded",
                      &gobot::VelocityCommandDebug3D::IsPolicyLoaded,
                      &gobot::VelocityCommandDebug3D::SetPolicyLoaded)(editor_only())
            .property("input_focused",
                      &gobot::VelocityCommandDebug3D::IsInputFocused,
                      &gobot::VelocityCommandDebug3D::SetInputFocused)(editor_only())
            .property("action_norm",
                      &gobot::VelocityCommandDebug3D::GetActionNorm,
                      &gobot::VelocityCommandDebug3D::SetActionNorm)(editor_only());

};
