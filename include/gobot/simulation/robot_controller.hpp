/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gobot/core/ref_counted.hpp"
#include "gobot/physics/physics_world.hpp"

namespace gobot {

class SimulationScene;

class GOBOT_EXPORT RobotController {
public:
    RobotController() = default;

    bool IsValid() const;

    const std::string& GetLastError() const;

    bool SetJointPositionTarget(const std::string& joint_name, RealType target_position);

    bool SetJointPositionTargets(const std::vector<std::string>& joint_names,
                                 const std::vector<RealType>& target_positions);

    bool SetJointVelocityTarget(const std::string& joint_name, RealType target_velocity);

    bool SetJointEffortTarget(const std::string& joint_name, RealType target_effort);

    bool SetJointPassive(const std::string& joint_name);

    bool ResetJointState(const std::string& joint_name, RealType position, RealType velocity = 0.0);

    bool ResetEnvironmentJointState(std::size_t environment_index,
                                    const std::string& joint_name,
                                    RealType position,
                                    RealType velocity = 0.0);

    bool SetEnvironmentJointPositionTarget(std::size_t environment_index,
                                           const std::string& joint_name,
                                           RealType target_position);

    bool SetNormalizedJointPositionTargets(const std::vector<RealType>& action);

    bool SetNormalizedJointPositionTargets(const std::vector<std::string>& joint_names,
                                           const std::vector<RealType>& action);

private:
    friend class SimulationScene;

    struct SessionState {
        std::atomic<std::uint64_t> generation{0};
    };

    RobotController(Ref<PhysicsWorld> world,
                    std::string entity_name,
                    std::weak_ptr<const SessionState> session_state,
                    std::uint64_t session_generation);

    bool EnsureReady();

    bool SetJointControl(const std::string& joint_name,
                         PhysicsJointControlMode control_mode,
                         RealType target);

    bool SetEnvironmentJointControl(std::size_t environment_index,
                                    const std::string& joint_name,
                                    PhysicsJointControlMode control_mode,
                                    RealType target);

    void SetLastError(std::string error);

    Ref<PhysicsWorld> world_;
    std::string entity_name_;
    std::weak_ptr<const SessionState> session_state_;
    std::uint64_t session_generation_{0};
    std::string last_error_;
};

} // namespace gobot
