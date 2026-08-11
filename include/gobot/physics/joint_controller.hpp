/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "gobot/core/math/geometry.hpp"

namespace gobot {

enum class PhysicsJointControlMode {
    Passive,
    Position,
    Velocity,
    Effort
};

struct JointControllerGains {
    RealType position_stiffness{0.0};
    RealType velocity_damping{0.0};
    RealType integral_gain{0.0};
    RealType integral_limit{0.0};
};

struct JointControllerLimits {
    bool has_position_limits{false};
    RealType lower_position_limit{0.0};
    RealType upper_position_limit{0.0};
    RealType effort_limit{0.0};
};

struct JointControllerCommand {
    PhysicsJointControlMode mode{PhysicsJointControlMode::Passive};
    RealType target_position{0.0};
    RealType target_velocity{0.0};
    RealType target_effort{0.0};
};

struct JointControllerState {
    RealType position{0.0};
    RealType velocity{0.0};
};

struct JointActuatorModel {
    std::uint32_t command_delay_steps{0};
    RealType command_deadband{0.0};
    RealType command_slew_rate{0.0};
    RealType strength_scale{1.0};
    RealType motor_velocity_limit{0.0};
    RealType motor_stall_effort{0.0};
};

struct JointControllerTelemetry {
    RealType commanded_target{0.0};
    RealType applied_target{0.0};
    RealType tracking_error{0.0};
    RealType applied_effort{0.0};
    bool delayed{false};
    bool deadband_applied{false};
    bool rate_limited{false};
    bool effort_saturated{false};
};

struct JointControllerRuntimeState {
    RealType integral_error{0.0};
    std::vector<JointControllerCommand> delayed_commands;
    PhysicsJointControlMode previous_mode{PhysicsJointControlMode::Passive};
    RealType previous_target{0.0};
    bool has_previous_target{false};
    JointControllerTelemetry telemetry;
};

struct PhysicsJointSnapshot;
struct PhysicsJointState;

class GOBOT_EXPORT JointController {
public:
    JointController() = default;

    explicit JointController(JointControllerGains gains);

    void SetGains(const JointControllerGains& gains);

    const JointControllerGains& GetGains() const;

    void SetActuatorModel(const JointActuatorModel& model);

    const JointActuatorModel& GetActuatorModel() const;

    const JointControllerTelemetry& GetTelemetry() const;

    JointControllerRuntimeState CaptureRuntimeState() const;

    void RestoreRuntimeState(const JointControllerRuntimeState& state);

    void Reset();

    RealType GetIntegralError() const;

    void RestoreIntegralError(RealType integral_error);

    JointControllerCommand ProcessCommand(const JointControllerState& state,
                                          const JointControllerCommand& command,
                                          RealType delta_time);

    RealType ApplyEffortEnvelope(RealType effort,
                                 const JointControllerState& state,
                                 const JointControllerLimits& limits);

    RealType GetStrengthScaleAtVelocity(RealType velocity) const;

    RealType ComputeEffort(const JointControllerState& state,
                           const JointControllerCommand& command,
                           const JointControllerLimits& limits,
                           RealType delta_time);

    static RealType ClampTargetPosition(RealType target_position, const JointControllerLimits& limits);

    static RealType ClampEffort(RealType effort, RealType effort_limit);

    static RealType MapNormalizedActionToTargetPosition(RealType normalized_action,
                                                        const JointControllerLimits& limits,
                                                        RealType fallback_center = 0.0,
                                                        RealType fallback_range = 1.0);

private:
    JointControllerGains gains_;
    JointActuatorModel actuator_model_;
    JointControllerTelemetry telemetry_;
    RealType integral_error_{0.0};
    std::deque<JointControllerCommand> delayed_commands_;
    PhysicsJointControlMode previous_mode_{PhysicsJointControlMode::Passive};
    RealType previous_target_{0.0};
    bool has_previous_target_{false};
};

JointControllerLimits MakeJointControllerLimits(const PhysicsJointSnapshot& joint_snapshot);

JointControllerState MakeJointControllerState(const PhysicsJointState& joint_state);

JointControllerCommand MakeJointControllerCommand(const PhysicsJointState& joint_state);

} // namespace gobot
