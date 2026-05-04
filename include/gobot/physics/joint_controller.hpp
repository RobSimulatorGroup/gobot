/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/physics/physics_types.hpp"

namespace gobot {

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

class GOBOT_EXPORT JointController {
public:
    JointController() = default;

    explicit JointController(JointControllerGains gains);

    void SetGains(const JointControllerGains& gains);

    const JointControllerGains& GetGains() const;

    void Reset();

    RealType GetIntegralError() const;

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
    RealType integral_error_{0.0};
};

JointControllerLimits MakeJointControllerLimits(const PhysicsJointSnapshot& joint_snapshot);

JointControllerState MakeJointControllerState(const PhysicsJointState& joint_state);

JointControllerCommand MakeJointControllerCommand(const PhysicsJointState& joint_state);

} // namespace gobot
