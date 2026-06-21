/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "gobot/simulation/robot_controller.hpp"
#include "gobot/simulation/simulation_entity.hpp"

namespace gobot {

class Node;

class GOBOT_EXPORT SimulationScene {
public:
    SimulationScene() = default;

    bool Initialize(Ref<PhysicsWorld> world, const Node* scene_root);

    void Clear();

    bool IsValid() const;

    const Node* GetSceneRoot() const;

    std::size_t GetEntityCount() const;

    const std::vector<SimulationEntity>& GetEntities() const;

    const SimulationEntity* GetEntity(const std::string& entity_name) const;

    RobotController GetRobotController(const std::string& entity_name) const;

    bool ResetEnvironment(std::size_t environment_index);

    bool StepEnvironment(std::size_t environment_index, RealType delta_time);

    bool StepEnvironmentBatch(RealType delta_time,
                              std::uint64_t ticks = 1,
                              std::size_t worker_count = 0);

    bool SetJointPositionTarget(const std::string& robot_name,
                                const std::string& joint_name,
                                RealType target_position);

    bool SetJointPositionTargets(const std::string& robot_name,
                                 const std::vector<std::string>& joint_names,
                                 const std::vector<RealType>& target_positions);

    bool SetJointVelocityTarget(const std::string& robot_name,
                                const std::string& joint_name,
                                RealType target_velocity);

    bool SetJointEffortTarget(const std::string& robot_name,
                              const std::string& joint_name,
                              RealType target_effort);

    bool SetJointPassive(const std::string& robot_name, const std::string& joint_name);

    bool ResetJointState(const std::string& robot_name,
                         const std::string& joint_name,
                         RealType position,
                         RealType velocity);

    bool ResetEnvironmentJointState(std::size_t environment_index,
                                    const std::string& robot_name,
                                    const std::string& joint_name,
                                    RealType position,
                                    RealType velocity);

    bool ResetLinkState(const std::string& robot_name,
                        const std::string& link_name,
                        const Vector3& position,
                        const Quaternion& orientation,
                        const Vector3& linear_velocity,
                        const Vector3& angular_velocity);

    bool ResetEnvironmentLinkState(std::size_t environment_index,
                                   const std::string& robot_name,
                                   const std::string& link_name,
                                   const Vector3& position,
                                   const Quaternion& orientation,
                                   const Vector3& linear_velocity,
                                   const Vector3& angular_velocity);

    bool ResetEnvironmentRobotStates(const std::vector<PhysicsEnvironmentRobotResetState>& reset_states);

    bool SetEnvironmentJointPositionTarget(std::size_t environment_index,
                                           const std::string& robot_name,
                                           const std::string& joint_name,
                                           RealType target_position);

    bool SetEnvironmentJointPositionTargets(const std::string& robot_name,
                                            const std::vector<std::string>& joint_names,
                                            const std::vector<RealType>& target_positions,
                                            std::size_t environment_count);

    bool SetLinkExternalForce(const std::string& robot_name,
                              const std::string& link_name,
                              const Vector3& point,
                              const Vector3& force);

    bool SetLinkSpringForce(const std::string& robot_name,
                            const std::string& link_name,
                            const Vector3& local_point,
                            const Vector3& target_point,
                            const Vector3& force_hint);

    void ClearExternalForces();

    bool SetRobotJointPositionTargetsFromNormalizedAction(const std::string& robot_name,
                                                          const std::vector<RealType>& action);

    bool SetRobotJointPositionTargetsFromNormalizedAction(const std::string& robot_name,
                                                          const std::vector<std::string>& joint_names,
                                                          const std::vector<RealType>& action);

    const std::string& GetLastError() const;

private:
    bool EnsureReady();

    RobotController GetRequiredRobotController(const std::string& robot_name);

    void SetLastError(std::string error);

    Ref<PhysicsWorld> world_;
    const Node* scene_root_{nullptr};
    std::vector<SimulationEntity> entities_;
    std::unordered_map<std::string, std::size_t> entity_indices_;
    std::string last_error_;
};

} // namespace gobot
