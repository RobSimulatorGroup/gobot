/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/joint_controller.hpp"

#include <algorithm>
#include <cmath>

#include "gobot/physics/physics_types.hpp"
#include "gobot/core/registration.hpp"

namespace gobot {
namespace {

RealType ClampValue(RealType value, RealType lower, RealType upper) {
    return std::min(std::max(value, lower), upper);
}

} // namespace

JointController::JointController(JointControllerGains gains)
    : gains_(gains) {
}

void JointController::SetGains(const JointControllerGains& gains) {
    gains_ = gains;
}

const JointControllerGains& JointController::GetGains() const {
    return gains_;
}

void JointController::Reset() {
    integral_error_ = 0.0;
}

RealType JointController::GetIntegralError() const {
    return integral_error_;
}

RealType JointController::ComputeEffort(const JointControllerState& state,
                                        const JointControllerCommand& command,
                                        const JointControllerLimits& limits,
                                        RealType delta_time) {
    switch (command.mode) {
        case PhysicsJointControlMode::Passive:
            return 0.0;
        case PhysicsJointControlMode::Effort:
            return ClampEffort(command.target_effort, limits.effort_limit);
        case PhysicsJointControlMode::Velocity: {
            const RealType velocity_error = command.target_velocity - state.velocity;
            return ClampEffort(gains_.velocity_damping * velocity_error, limits.effort_limit);
        }
        case PhysicsJointControlMode::Position:
            break;
    }

    const RealType target_position = ClampTargetPosition(command.target_position, limits);
    const RealType position_error = target_position - state.position;

    if (delta_time > 0.0 && gains_.integral_gain != 0.0) {
        integral_error_ += position_error * delta_time;
        if (gains_.integral_limit > 0.0) {
            integral_error_ = ClampValue(integral_error_, -gains_.integral_limit, gains_.integral_limit);
        }
    }

    const RealType effort =
            gains_.position_stiffness * position_error -
            gains_.velocity_damping * state.velocity +
            gains_.integral_gain * integral_error_;
    return ClampEffort(effort, limits.effort_limit);
}

RealType JointController::ClampTargetPosition(RealType target_position, const JointControllerLimits& limits) {
    if (!limits.has_position_limits) {
        return target_position;
    }

    return ClampValue(target_position, limits.lower_position_limit, limits.upper_position_limit);
}

RealType JointController::ClampEffort(RealType effort, RealType effort_limit) {
    if (effort_limit <= 0.0) {
        return effort;
    }

    return ClampValue(effort, -effort_limit, effort_limit);
}

RealType JointController::MapNormalizedActionToTargetPosition(RealType normalized_action,
                                                             const JointControllerLimits& limits,
                                                             RealType fallback_center,
                                                             RealType fallback_range) {
    const RealType action = ClampValue(normalized_action, -1.0, 1.0);
    if (limits.has_position_limits) {
        const RealType center = (limits.lower_position_limit + limits.upper_position_limit) * 0.5;
        const RealType range = (limits.upper_position_limit - limits.lower_position_limit) * 0.5;
        return center + action * range;
    }

    const RealType range = std::max<RealType>(0.0, fallback_range);
    return fallback_center + action * range;
}

JointControllerLimits MakeJointControllerLimits(const PhysicsJointSnapshot& joint_snapshot) {
    JointControllerLimits limits;
    limits.lower_position_limit = joint_snapshot.lower_limit;
    limits.upper_position_limit = joint_snapshot.upper_limit;
    limits.effort_limit = std::max<RealType>(0.0, joint_snapshot.effort_limit);
    limits.has_position_limits = limits.upper_position_limit > limits.lower_position_limit;
    return limits;
}

JointControllerState MakeJointControllerState(const PhysicsJointState& joint_state) {
    JointControllerState state;
    state.position = joint_state.position;
    state.velocity = joint_state.velocity;
    return state;
}

JointControllerCommand MakeJointControllerCommand(const PhysicsJointState& joint_state) {
    JointControllerCommand command;
    command.mode = joint_state.control_mode;
    command.target_position = joint_state.target_position;
    command.target_velocity = joint_state.target_velocity;
    command.target_effort = joint_state.target_effort;
    return command;
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<JointControllerGains>("JointControllerGains")
            .constructor()
            .property("position_stiffness", &JointControllerGains::position_stiffness)
            .property("velocity_damping", &JointControllerGains::velocity_damping)
            .property("integral_gain", &JointControllerGains::integral_gain)
            .property("integral_limit", &JointControllerGains::integral_limit);

};
