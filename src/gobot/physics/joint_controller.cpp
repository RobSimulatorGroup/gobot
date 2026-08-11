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

void JointController::SetActuatorModel(const JointActuatorModel& model) {
    actuator_model_ = model;
}

const JointActuatorModel& JointController::GetActuatorModel() const {
    return actuator_model_;
}

const JointControllerTelemetry& JointController::GetTelemetry() const {
    return telemetry_;
}

JointControllerRuntimeState JointController::CaptureRuntimeState() const {
    JointControllerRuntimeState state;
    state.integral_error = integral_error_;
    state.delayed_commands.assign(delayed_commands_.begin(), delayed_commands_.end());
    state.previous_mode = previous_mode_;
    state.previous_target = previous_target_;
    state.has_previous_target = has_previous_target_;
    state.telemetry = telemetry_;
    return state;
}

void JointController::RestoreRuntimeState(const JointControllerRuntimeState& state) {
    RestoreIntegralError(state.integral_error);
    delayed_commands_.assign(state.delayed_commands.begin(), state.delayed_commands.end());
    previous_mode_ = state.previous_mode;
    previous_target_ = state.previous_target;
    has_previous_target_ = state.has_previous_target;
    telemetry_ = state.telemetry;
}

void JointController::Reset() {
    integral_error_ = 0.0;
    delayed_commands_.clear();
    previous_mode_ = PhysicsJointControlMode::Passive;
    previous_target_ = 0.0;
    has_previous_target_ = false;
    telemetry_ = {};
}

JointControllerCommand JointController::ProcessCommand(
        const JointControllerState& state,
        const JointControllerCommand& command,
        RealType delta_time) {
    const auto target_for = [](const JointControllerCommand& value) {
        switch (value.mode) {
            case PhysicsJointControlMode::Position: return value.target_position;
            case PhysicsJointControlMode::Velocity: return value.target_velocity;
            case PhysicsJointControlMode::Effort: return value.target_effort;
            case PhysicsJointControlMode::Passive: return RealType{0.0};
        }
        return RealType{0.0};
    };
    const auto set_target = [](JointControllerCommand* value, RealType target) {
        switch (value->mode) {
            case PhysicsJointControlMode::Position: value->target_position = target; break;
            case PhysicsJointControlMode::Velocity: value->target_velocity = target; break;
            case PhysicsJointControlMode::Effort: value->target_effort = target; break;
            case PhysicsJointControlMode::Passive: break;
        }
    };

    telemetry_ = {};
    telemetry_.commanded_target = target_for(command);
    delayed_commands_.push_back(command);
    JointControllerCommand processed = command;
    if (delayed_commands_.size() <= actuator_model_.command_delay_steps) {
        telemetry_.delayed = true;
        processed.mode = command.mode;
        if (command.mode == PhysicsJointControlMode::Position) {
            processed.target_position = state.position;
        } else if (command.mode == PhysicsJointControlMode::Velocity) {
            processed.target_velocity = state.velocity;
        } else if (command.mode == PhysicsJointControlMode::Effort) {
            processed.target_effort = 0.0;
        }
    } else {
        processed = delayed_commands_.front();
        delayed_commands_.pop_front();
        telemetry_.delayed = actuator_model_.command_delay_steps > 0;
    }

    RealType target = target_for(processed);
    if (actuator_model_.command_deadband > 0.0) {
        RealType reference = 0.0;
        if (processed.mode == PhysicsJointControlMode::Position) {
            reference = state.position;
        } else if (processed.mode == PhysicsJointControlMode::Velocity) {
            reference = state.velocity;
        }
        if (std::abs(target - reference) < actuator_model_.command_deadband) {
            target = reference;
            telemetry_.deadband_applied = true;
        }
    }

    if (!has_previous_target_ || previous_mode_ != processed.mode) {
        if (processed.mode == PhysicsJointControlMode::Position) {
            previous_target_ = state.position;
        } else if (processed.mode == PhysicsJointControlMode::Velocity) {
            previous_target_ = state.velocity;
        } else {
            previous_target_ = 0.0;
        }
        has_previous_target_ = true;
    }
    if (actuator_model_.command_slew_rate > 0.0 && delta_time > 0.0) {
        const RealType maximum_delta = actuator_model_.command_slew_rate * delta_time;
        const RealType limited = ClampValue(
                target, previous_target_ - maximum_delta, previous_target_ + maximum_delta);
        telemetry_.rate_limited = limited != target;
        target = limited;
    }
    previous_mode_ = processed.mode;
    previous_target_ = target;
    set_target(&processed, target);
    telemetry_.applied_target = target;
    if (processed.mode == PhysicsJointControlMode::Position) {
        telemetry_.tracking_error = target - state.position;
    } else if (processed.mode == PhysicsJointControlMode::Velocity) {
        telemetry_.tracking_error = target - state.velocity;
    }
    return processed;
}

RealType JointController::GetStrengthScaleAtVelocity(RealType velocity) const {
    RealType scale = std::max<RealType>(0.0, actuator_model_.strength_scale);
    if (actuator_model_.motor_velocity_limit > 0.0) {
        scale *= std::max<RealType>(
                0.0, 1.0 - std::abs(velocity) / actuator_model_.motor_velocity_limit);
    }
    return scale;
}

RealType JointController::ApplyEffortEnvelope(
        RealType effort,
        const JointControllerState& state,
        const JointControllerLimits& limits) {
    const RealType scaled_effort = effort *
                                   std::max<RealType>(0.0, actuator_model_.strength_scale);
    RealType effort_limit = std::max<RealType>(0.0, limits.effort_limit);
    if (actuator_model_.motor_stall_effort > 0.0) {
        const RealType motor_limit = actuator_model_.motor_stall_effort *
                                     (actuator_model_.motor_velocity_limit > 0.0
                                              ? std::max<RealType>(
                                                        0.0,
                                                        1.0 - std::abs(state.velocity) /
                                                                      actuator_model_.motor_velocity_limit)
                                              : 1.0);
        effort_limit = effort_limit > 0.0 ? std::min(effort_limit, motor_limit) : motor_limit;
    }
    const RealType result = ClampEffort(scaled_effort, effort_limit);
    telemetry_.applied_effort = result;
    telemetry_.effort_saturated = result != scaled_effort;
    return result;
}

RealType JointController::GetIntegralError() const {
    return integral_error_;
}

void JointController::RestoreIntegralError(RealType integral_error) {
    integral_error_ = std::isfinite(integral_error) ? integral_error : 0.0;
    if (gains_.integral_limit > 0.0) {
        integral_error_ = ClampValue(
                integral_error_, -gains_.integral_limit, gains_.integral_limit);
    }
}

RealType JointController::ComputeEffort(const JointControllerState& state,
                                        const JointControllerCommand& command,
                                        const JointControllerLimits& limits,
                                        RealType delta_time) {
    const JointControllerCommand processed = ProcessCommand(state, command, delta_time);
    switch (processed.mode) {
        case PhysicsJointControlMode::Passive:
            return 0.0;
        case PhysicsJointControlMode::Effort:
            return ApplyEffortEnvelope(processed.target_effort, state, limits);
        case PhysicsJointControlMode::Velocity: {
            const RealType velocity_error = processed.target_velocity - state.velocity;
            return ApplyEffortEnvelope(gains_.velocity_damping * velocity_error, state, limits);
        }
        case PhysicsJointControlMode::Position:
            break;
    }

    const RealType target_position = ClampTargetPosition(processed.target_position, limits);
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
    return ApplyEffortEnvelope(effort, state, limits);
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
