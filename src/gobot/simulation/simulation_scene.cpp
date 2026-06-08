/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/simulation/simulation_scene.hpp"

#include <utility>

#include <fmt/format.h>

namespace gobot {

bool SimulationScene::Initialize(Ref<PhysicsWorld> world, const Node* scene_root) {
    Clear();
    if (!world.IsValid()) {
        SetLastError("Cannot initialize SimulationScene without a physics world.");
        return false;
    }

    world_ = std::move(world);
    scene_root_ = scene_root;
    const PhysicsSceneSnapshot& snapshot = world_->GetSceneSnapshot();
    entities_.reserve(snapshot.robots.size());
    for (const PhysicsRobotSnapshot& robot_snapshot : snapshot.robots) {
        entity_indices_[robot_snapshot.name] = entities_.size();
        entities_.emplace_back(robot_snapshot);
    }

    last_error_.clear();
    return true;
}

void SimulationScene::Clear() {
    world_.Reset();
    scene_root_ = nullptr;
    entities_.clear();
    entity_indices_.clear();
    last_error_.clear();
}

bool SimulationScene::IsValid() const {
    return world_.IsValid();
}

const Node* SimulationScene::GetSceneRoot() const {
    return scene_root_;
}

std::size_t SimulationScene::GetEntityCount() const {
    return entities_.size();
}

const std::vector<SimulationEntity>& SimulationScene::GetEntities() const {
    return entities_;
}

const SimulationEntity* SimulationScene::GetEntity(const std::string& entity_name) const {
    const auto iter = entity_indices_.find(entity_name);
    if (iter == entity_indices_.end() || iter->second >= entities_.size()) {
        return nullptr;
    }

    return &entities_[iter->second];
}

RobotController SimulationScene::GetRobotController(const std::string& entity_name) const {
    return RobotController(world_, GetEntity(entity_name));
}

bool SimulationScene::ResetEnvironment(std::size_t environment_index) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->ResetEnvironment(environment_index)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationScene::StepEnvironment(std::size_t environment_index, RealType delta_time) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->StepEnvironment(environment_index, delta_time)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationScene::StepEnvironmentBatch(RealType delta_time, std::uint64_t ticks, std::size_t worker_count) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->StepEnvironmentBatch(delta_time, ticks, worker_count)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationScene::SetJointPositionTarget(const std::string& robot_name,
                                             const std::string& joint_name,
                                             RealType target_position) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.SetJointPositionTarget(joint_name, target_position)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::SetJointVelocityTarget(const std::string& robot_name,
                                             const std::string& joint_name,
                                             RealType target_velocity) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.SetJointVelocityTarget(joint_name, target_velocity)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::SetJointEffortTarget(const std::string& robot_name,
                                           const std::string& joint_name,
                                           RealType target_effort) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.SetJointEffortTarget(joint_name, target_effort)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::SetJointPassive(const std::string& robot_name, const std::string& joint_name) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.SetJointPassive(joint_name)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::ResetJointState(const std::string& robot_name,
                                      const std::string& joint_name,
                                      RealType position,
                                      RealType velocity) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.ResetJointState(joint_name, position, velocity)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::ResetEnvironmentJointState(std::size_t environment_index,
                                                 const std::string& robot_name,
                                                 const std::string& joint_name,
                                                 RealType position,
                                                 RealType velocity) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.ResetEnvironmentJointState(environment_index, joint_name, position, velocity)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::ResetLinkState(const std::string& robot_name,
                                     const std::string& link_name,
                                     const Vector3& position,
                                     const Quaternion& orientation,
                                     const Vector3& linear_velocity,
                                     const Vector3& angular_velocity) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->ResetLinkState(robot_name,
                                link_name,
                                position,
                                orientation,
                                linear_velocity,
                                angular_velocity)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationScene::ResetEnvironmentLinkState(std::size_t environment_index,
                                                const std::string& robot_name,
                                                const std::string& link_name,
                                                const Vector3& position,
                                                const Quaternion& orientation,
                                                const Vector3& linear_velocity,
                                                const Vector3& angular_velocity) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->ResetEnvironmentLinkState(environment_index,
                                           robot_name,
                                           link_name,
                                           position,
                                           orientation,
                                           linear_velocity,
                                           angular_velocity)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationScene::SetEnvironmentJointPositionTarget(std::size_t environment_index,
                                                        const std::string& robot_name,
                                                        const std::string& joint_name,
                                                        RealType target_position) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.SetEnvironmentJointPositionTarget(environment_index, joint_name, target_position)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::SetEnvironmentJointPositionTargets(const std::string& robot_name,
                                                         const std::vector<std::string>& joint_names,
                                                         const std::vector<RealType>& target_positions,
                                                         std::size_t environment_count) {
    if (!EnsureReady()) {
        return false;
    }
    if (GetEntity(robot_name) == nullptr) {
        SetLastError(fmt::format("Simulation scene has no entity named '{}'.", robot_name));
        return false;
    }
    if (!world_->SetEnvironmentJointControls(robot_name,
                                             joint_names,
                                             PhysicsJointControlMode::Position,
                                             target_positions,
                                             environment_count)) {
        SetLastError(world_->GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::SetLinkExternalForce(const std::string& robot_name,
                                           const std::string& link_name,
                                           const Vector3& point,
                                           const Vector3& force) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->SetLinkExternalForce(robot_name, link_name, point, force)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool SimulationScene::SetLinkSpringForce(const std::string& robot_name,
                                         const std::string& link_name,
                                         const Vector3& local_point,
                                         const Vector3& target_point,
                                         const Vector3& force_hint) {
    if (!EnsureReady()) {
        return false;
    }

    if (!world_->SetLinkSpringForce(robot_name, link_name, local_point, target_point, force_hint)) {
        SetLastError(world_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

void SimulationScene::ClearExternalForces() {
    if (world_.IsValid()) {
        world_->ClearExternalForces();
    }
}

bool SimulationScene::SetRobotJointPositionTargetsFromNormalizedAction(const std::string& robot_name,
                                                                       const std::vector<RealType>& action) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.SetNormalizedJointPositionTargets(action)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

bool SimulationScene::SetRobotJointPositionTargetsFromNormalizedAction(
        const std::string& robot_name,
        const std::vector<std::string>& joint_names,
        const std::vector<RealType>& action) {
    RobotController controller = GetRequiredRobotController(robot_name);
    if (!controller.IsValid()) {
        return false;
    }
    if (!controller.SetNormalizedJointPositionTargets(joint_names, action)) {
        SetLastError(controller.GetLastError());
        return false;
    }
    last_error_.clear();
    return true;
}

const std::string& SimulationScene::GetLastError() const {
    return last_error_;
}

bool SimulationScene::EnsureReady() {
    if (!world_.IsValid()) {
        SetLastError("Simulation scene has not been initialized with a physics world.");
        return false;
    }

    if (!world_->IsAvailable()) {
        SetLastError(world_->GetLastError());
        return false;
    }

    return true;
}

RobotController SimulationScene::GetRequiredRobotController(const std::string& robot_name) {
    if (!EnsureReady()) {
        return {};
    }

    const SimulationEntity* entity = GetEntity(robot_name);
    if (entity == nullptr) {
        SetLastError(fmt::format("Simulation scene has no entity named '{}'.", robot_name));
        return {};
    }

    return RobotController(world_, entity);
}

void SimulationScene::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot
