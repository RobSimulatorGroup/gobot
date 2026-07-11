/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/robot_controller.hpp"

#include <algorithm>
#include <utility>

#include <fmt/format.h>

#include "gobot/core/robotics_types.hpp"
#include "gobot/physics/joint_controller.hpp"
#include "gobot/simulation/simulation_entity.hpp"

namespace gobot {
namespace {

const PhysicsRobotSnapshot* FindRobotSnapshot(const PhysicsSceneSnapshot& snapshot,
                                              const std::string& robot_name) {
    for (const PhysicsRobotSnapshot& robot_snapshot : snapshot.robots) {
        if (robot_snapshot.name == robot_name) {
            return &robot_snapshot;
        }
    }
    return nullptr;
}

const PhysicsRobotState* FindRobotState(const PhysicsSceneState& state,
                                        const std::string& robot_name) {
    for (const PhysicsRobotState& robot_state : state.robots) {
        if (robot_state.name == robot_name) {
            return &robot_state;
        }
    }
    return nullptr;
}

} // namespace

RobotController::RobotController(Ref<PhysicsWorld> world, const SimulationEntity* entity)
    : world_(std::move(world)),
      entity_(entity) {
}

bool RobotController::IsValid() const {
    return world_.IsValid() && entity_ != nullptr;
}

const std::string& RobotController::GetLastError() const {
    return last_error_;
}

bool RobotController::SetJointPositionTarget(const std::string& joint_name, RealType target_position) {
    return SetJointControl(joint_name, PhysicsJointControlMode::Position, target_position);
}

bool RobotController::SetJointPositionTargets(const std::vector<std::string>& joint_names,
                                              const std::vector<RealType>& target_positions) {
    if (!EnsureReady()) {
        return false;
    }
    if (joint_names.size() != target_positions.size()) {
        SetLastError(fmt::format("Robot '{}' expected {} joint position target value(s), got {}.",
                                 entity_->GetName(),
                                 joint_names.size(),
                                 target_positions.size()));
        return false;
    }
    for (std::size_t index = 0; index < joint_names.size(); ++index) {
        if (!world_->SetJointControl(entity_->GetName(),
                                     joint_names[index],
                                     PhysicsJointControlMode::Position,
                                     target_positions[index])) {
            SetLastError(world_->GetLastError());
            return false;
        }
    }

    last_error_.clear();
    return true;
}

bool RobotController::SetJointVelocityTarget(const std::string& joint_name, RealType target_velocity) {
    return SetJointControl(joint_name, PhysicsJointControlMode::Velocity, target_velocity);
}

bool RobotController::SetJointEffortTarget(const std::string& joint_name, RealType target_effort) {
    return SetJointControl(joint_name, PhysicsJointControlMode::Effort, target_effort);
}

bool RobotController::SetJointPassive(const std::string& joint_name) {
    return SetJointControl(joint_name, PhysicsJointControlMode::Passive, 0.0);
}

bool RobotController::ResetJointState(const std::string& joint_name, RealType position, RealType velocity) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->ResetJointState(entity_->GetName(), joint_name, position, velocity)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool RobotController::ResetEnvironmentJointState(std::size_t environment_index,
                                                 const std::string& joint_name,
                                                 RealType position,
                                                 RealType velocity) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->ResetEnvironmentJointState(environment_index, entity_->GetName(), joint_name, position, velocity)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool RobotController::SetEnvironmentJointPositionTarget(std::size_t environment_index,
                                                        const std::string& joint_name,
                                                        RealType target_position) {
    return SetEnvironmentJointControl(environment_index,
                                      joint_name,
                                      PhysicsJointControlMode::Position,
                                      target_position);
}

bool RobotController::SetNormalizedJointPositionTargets(const std::vector<RealType>& action) {
    if (!EnsureReady()) {
        return false;
    }

    const PhysicsRobotSnapshot* robot_snapshot =
            FindRobotSnapshot(world_->GetSceneSnapshot(), entity_->GetName());
    const PhysicsRobotState* robot_state =
            FindRobotState(world_->GetSceneState(), entity_->GetName());
    if (robot_snapshot == nullptr || robot_state == nullptr) {
        SetLastError(fmt::format("Cannot set normalized action for missing robot '{}'.", entity_->GetName()));
        return false;
    }

    std::size_t action_index = 0;
    for (std::size_t joint_index = 0; joint_index < robot_snapshot->joints.size(); ++joint_index) {
        if (joint_index >= robot_state->joints.size()) {
            continue;
        }

        const PhysicsJointSnapshot& joint_snapshot = robot_snapshot->joints[joint_index];
        const PhysicsJointState& joint_state = robot_state->joints[joint_index];
        const auto joint_type = static_cast<JointType>(joint_snapshot.joint_type);
        if (joint_type != JointType::Revolute &&
            joint_type != JointType::Continuous &&
            joint_type != JointType::Prismatic) {
            continue;
        }

        if (action_index >= action.size()) {
            SetLastError(fmt::format("Robot '{}' expected at least {} joint action value(s), got {}.",
                                     entity_->GetName(),
                                     action_index + 1,
                                     action.size()));
            return false;
        }

        JointControllerLimits limits = MakeJointControllerLimits(joint_snapshot);
        if (joint_type == JointType::Continuous) {
            limits.has_position_limits = false;
        }

        const RealType target_position =
                JointController::MapNormalizedActionToTargetPosition(action[action_index],
                                                                     limits,
                                                                     joint_state.position,
                                                                     1.0);
        if (!world_->SetJointControl(entity_->GetName(),
                                     joint_state.joint_name,
                                     PhysicsJointControlMode::Position,
                                     target_position)) {
            SetLastError(world_->GetLastError());
            return false;
        }

        ++action_index;
    }

    if (action_index < action.size()) {
        SetLastError(fmt::format("Robot '{}' expected {} joint action value(s), got {}.",
                                 entity_->GetName(),
                                 action_index,
                                 action.size()));
        return false;
    }

    last_error_.clear();
    return true;
}

bool RobotController::SetNormalizedJointPositionTargets(const std::vector<std::string>& joint_names,
                                                        const std::vector<RealType>& action) {
    if (!EnsureReady()) {
        return false;
    }

    if (joint_names.size() != action.size()) {
        SetLastError(fmt::format("Robot '{}' expected {} named joint action value(s), got {}.",
                                 entity_->GetName(),
                                 joint_names.size(),
                                 action.size()));
        return false;
    }

    const PhysicsRobotSnapshot* robot_snapshot =
            FindRobotSnapshot(world_->GetSceneSnapshot(), entity_->GetName());
    const PhysicsRobotState* robot_state =
            FindRobotState(world_->GetSceneState(), entity_->GetName());
    if (robot_snapshot == nullptr || robot_state == nullptr) {
        SetLastError(fmt::format("Cannot set normalized action for missing robot '{}'.", entity_->GetName()));
        return false;
    }

    for (std::size_t action_index = 0; action_index < joint_names.size(); ++action_index) {
        const std::string& joint_name = joint_names[action_index];
        const auto snapshot_iter = std::find_if(robot_snapshot->joints.begin(),
                                                robot_snapshot->joints.end(),
                                                [&joint_name](const PhysicsJointSnapshot& joint_snapshot) {
                                                    return joint_snapshot.name == joint_name;
                                                });
        const auto state_iter = std::find_if(robot_state->joints.begin(),
                                             robot_state->joints.end(),
                                             [&joint_name](const PhysicsJointState& joint_state) {
                                                 return joint_state.joint_name == joint_name;
                                             });
        if (snapshot_iter == robot_snapshot->joints.end() || state_iter == robot_state->joints.end()) {
            SetLastError(fmt::format("Robot '{}' has no joint named '{}'.", entity_->GetName(), joint_name));
            return false;
        }

        const auto joint_type = static_cast<JointType>(snapshot_iter->joint_type);
        if (joint_type != JointType::Revolute &&
            joint_type != JointType::Continuous &&
            joint_type != JointType::Prismatic) {
            SetLastError(fmt::format("Robot '{}' joint '{}' is not controllable by normalized position action.",
                                     entity_->GetName(),
                                     joint_name));
            return false;
        }

        JointControllerLimits limits = MakeJointControllerLimits(*snapshot_iter);
        if (joint_type == JointType::Continuous) {
            limits.has_position_limits = false;
        }

        const RealType target_position =
                JointController::MapNormalizedActionToTargetPosition(action[action_index],
                                                                     limits,
                                                                     state_iter->position,
                                                                     1.0);
        if (!world_->SetJointControl(entity_->GetName(),
                                     state_iter->joint_name,
                                     PhysicsJointControlMode::Position,
                                     target_position)) {
            SetLastError(world_->GetLastError());
            return false;
        }
    }

    last_error_.clear();
    return true;
}

bool RobotController::EnsureReady() {
    if (!world_.IsValid() || entity_ == nullptr) {
        SetLastError("Robot controller is not bound to a simulation world and entity.");
        return false;
    }

    if (!world_->IsAvailable()) {
        SetLastError(world_->GetLastError());
        return false;
    }

    return true;
}

bool RobotController::SetJointControl(const std::string& joint_name,
                                      PhysicsJointControlMode control_mode,
                                      RealType target) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->SetJointControl(entity_->GetName(), joint_name, control_mode, target)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool RobotController::SetEnvironmentJointControl(std::size_t environment_index,
                                                 const std::string& joint_name,
                                                 PhysicsJointControlMode control_mode,
                                                 RealType target) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->SetEnvironmentJointControl(environment_index,
                                            entity_->GetName(),
                                            joint_name,
                                            control_mode,
                                            target)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

void RobotController::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot
