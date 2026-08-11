/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/math_defs.hpp"

namespace gobot {

class GOBOT_EXPORT JointActuatorConfig : public Resource {
    GOBCLASS(JointActuatorConfig, Resource)

public:
    void SetCommandDelaySteps(std::uint32_t value);
    std::uint32_t GetCommandDelaySteps() const;

    void SetCommandDeadband(RealType value);
    RealType GetCommandDeadband() const;

    void SetCommandSlewRate(RealType value);
    RealType GetCommandSlewRate() const;

    void SetStrengthScale(RealType value);
    RealType GetStrengthScale() const;

    void SetMotorVelocityLimit(RealType value);
    RealType GetMotorVelocityLimit() const;

    void SetMotorStallEffort(RealType value);
    RealType GetMotorStallEffort() const;

private:
    std::uint32_t command_delay_steps_{0};
    RealType command_deadband_{0.0};
    RealType command_slew_rate_{0.0};
    RealType strength_scale_{1.0};
    RealType motor_velocity_limit_{0.0};
    RealType motor_stall_effort_{0.0};
};

} // namespace gobot
