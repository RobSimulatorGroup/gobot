/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/joint_actuator_config.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void JointActuatorConfig::SetCommandDelaySteps(std::uint32_t value) {
    command_delay_steps_ = value;
    MarkChanged();
}

std::uint32_t JointActuatorConfig::GetCommandDelaySteps() const { return command_delay_steps_; }

void JointActuatorConfig::SetCommandDeadband(RealType value) {
    command_deadband_ = value;
    MarkChanged();
}

RealType JointActuatorConfig::GetCommandDeadband() const { return command_deadband_; }

void JointActuatorConfig::SetCommandSlewRate(RealType value) {
    command_slew_rate_ = value;
    MarkChanged();
}

RealType JointActuatorConfig::GetCommandSlewRate() const { return command_slew_rate_; }

void JointActuatorConfig::SetStrengthScale(RealType value) {
    strength_scale_ = value;
    MarkChanged();
}

RealType JointActuatorConfig::GetStrengthScale() const { return strength_scale_; }

void JointActuatorConfig::SetMotorVelocityLimit(RealType value) {
    motor_velocity_limit_ = value;
    MarkChanged();
}

RealType JointActuatorConfig::GetMotorVelocityLimit() const { return motor_velocity_limit_; }

void JointActuatorConfig::SetMotorStallEffort(RealType value) {
    motor_stall_effort_ = value;
    MarkChanged();
}

RealType JointActuatorConfig::GetMotorStallEffort() const { return motor_stall_effort_; }

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::JointActuatorConfig>("JointActuatorConfig")
            .constructor()(CtorAsRawPtr)
            .property("command_delay_steps", &gobot::JointActuatorConfig::GetCommandDelaySteps,
                      &gobot::JointActuatorConfig::SetCommandDelaySteps)
            .property("command_deadband", &gobot::JointActuatorConfig::GetCommandDeadband,
                      &gobot::JointActuatorConfig::SetCommandDeadband)
            .property("command_slew_rate", &gobot::JointActuatorConfig::GetCommandSlewRate,
                      &gobot::JointActuatorConfig::SetCommandSlewRate)
            .property("strength_scale", &gobot::JointActuatorConfig::GetStrengthScale,
                      &gobot::JointActuatorConfig::SetStrengthScale)
            .property("motor_velocity_limit", &gobot::JointActuatorConfig::GetMotorVelocityLimit,
                      &gobot::JointActuatorConfig::SetMotorVelocityLimit)
            .property("motor_stall_effort", &gobot::JointActuatorConfig::GetMotorStallEffort,
                      &gobot::JointActuatorConfig::SetMotorStallEffort);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::JointActuatorConfig>, gobot::Ref<gobot::Resource>>();
}
