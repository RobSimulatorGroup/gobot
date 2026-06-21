/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/physics/physics_types.hpp"

namespace gobot {

class Node;

class GOBOT_EXPORT PhysicsWorld : public RefCounted {
    GOBCLASS(PhysicsWorld, RefCounted)

public:
    ~PhysicsWorld() override = default;

    virtual PhysicsBackendType GetBackendType() const = 0;

    virtual bool IsAvailable() const = 0;

    virtual const std::string& GetLastError() const = 0;

    const PhysicsWorldSettings& GetSettings() const;

    void SetSettings(const PhysicsWorldSettings& settings);

    virtual bool BuildFromScene(const Node* scene_root);

    virtual bool RestoreCompatibleState(const PhysicsSceneState& previous_state);

    virtual void Reset();

    virtual void Step(RealType delta_time);

    virtual bool ConfigureEnvironmentBatch(std::size_t environment_count);

    virtual std::size_t GetEnvironmentCount() const;

    virtual const PhysicsSceneState* GetEnvironmentState(std::size_t environment_index) const;

    virtual bool ResetEnvironment(std::size_t environment_index);

    virtual bool StepEnvironment(std::size_t environment_index, RealType delta_time);

    virtual bool StepEnvironmentBatch(RealType delta_time,
                                      std::uint64_t ticks = 1,
                                      std::size_t worker_count = 0);

    virtual std::size_t ResolveEnvironmentBatchWorkerCount(std::size_t worker_count) const;

    virtual bool ResetJointState(const std::string& robot_name,
                                 const std::string& joint_name,
                                 RealType position,
                                 RealType velocity = 0.0);

    virtual bool ResetEnvironmentJointState(std::size_t environment_index,
                                            const std::string& robot_name,
                                            const std::string& joint_name,
                                            RealType position,
                                            RealType velocity = 0.0);

    virtual bool ResetLinkState(const std::string& robot_name,
                                const std::string& link_name,
                                const Vector3& position,
                                const Quaternion& orientation = Quaternion::Identity(),
                                const Vector3& linear_velocity = Vector3::Zero(),
                                const Vector3& angular_velocity = Vector3::Zero());

    virtual bool ResetEnvironmentLinkState(std::size_t environment_index,
                                           const std::string& robot_name,
                                           const std::string& link_name,
                                           const Vector3& position,
                                           const Quaternion& orientation = Quaternion::Identity(),
                                           const Vector3& linear_velocity = Vector3::Zero(),
                                           const Vector3& angular_velocity = Vector3::Zero());

    virtual bool ResetEnvironmentRobotStates(const std::vector<PhysicsEnvironmentRobotResetState>& reset_states);

    bool SetJointControl(const std::string& robot_name,
                         const std::string& joint_name,
                         PhysicsJointControlMode control_mode,
                         RealType target);

    virtual bool SetEnvironmentJointControl(std::size_t environment_index,
                                            const std::string& robot_name,
                                            const std::string& joint_name,
                                            PhysicsJointControlMode control_mode,
                                            RealType target);

    virtual bool SetEnvironmentJointControls(const std::string& robot_name,
                                             const std::vector<std::string>& joint_names,
                                             PhysicsJointControlMode control_mode,
                                             const std::vector<RealType>& targets,
                                             std::size_t environment_count);

    virtual bool SetLinkExternalForce(const std::string& robot_name,
                                      const std::string& link_name,
                                      const Vector3& point,
                                      const Vector3& force);

    virtual bool SetLinkSpringForce(const std::string& robot_name,
                                    const std::string& link_name,
                                    const Vector3& local_point,
                                    const Vector3& target_point,
                                    const Vector3& force_hint);

    virtual void ClearExternalForces();

    const PhysicsSceneSnapshot& GetSceneSnapshot() const;

    const PhysicsSceneState& GetSceneState() const;

    virtual PhysicsRaycastHit RaycastTerrain(const PhysicsRaycastQuery& query) const;

protected:
    bool CaptureSceneSnapshot(const Node* scene_root);

    PhysicsRaycastHit RaycastTerrainFallback(const PhysicsRaycastQuery& query) const;

    virtual PhysicsRaycastHit RaycastTerrainForSensor(const PhysicsRaycastQuery& query,
                                                      std::size_t environment_index) const;

    void UpdateRaycastSensorState(PhysicsSensorState& sensor_state,
                                  const PhysicsSensorSnapshot& sensor_snapshot,
                                  const Affine3& parent_transform,
                                  RealType timestamp,
                                  std::size_t environment_index = 0);

    void UpdateSensorGlobalTransformsAndRaycastSensors(PhysicsSceneState& scene_state,
                                                       RealType timestamp,
                                                       std::size_t environment_index = 0);

    PhysicsJointState* FindJointState(const std::string& robot_name,
                                      const std::string& joint_name);

    PhysicsJointState* FindJointStateIn(PhysicsSceneState& scene_state,
                                        const std::string& robot_name,
                                        const std::string& joint_name);

    PhysicsLinkState* FindMutableLinkState(const std::string& robot_name,
                                           const std::string& link_name);

    PhysicsLinkState* FindMutableLinkStateIn(PhysicsSceneState& scene_state,
                                             const std::string& robot_name,
                                             const std::string& link_name);

    bool ResetJointStateIn(PhysicsSceneState& scene_state,
                           const std::string& robot_name,
                           const std::string& joint_name,
                           RealType position,
                           RealType velocity);

    bool ResetLinkStateIn(PhysicsSceneState& scene_state,
                          const std::string& robot_name,
                          const std::string& link_name,
                          const Vector3& position,
                          const Quaternion& orientation,
                          const Vector3& linear_velocity,
                          const Vector3& angular_velocity);

    bool SetJointControlIn(PhysicsSceneState& scene_state,
                           const std::string& robot_name,
                           const std::string& joint_name,
                           PhysicsJointControlMode control_mode,
                           RealType target);

    PhysicsSceneState MakeSceneStateFromSnapshot() const;

    void ResetSceneStateFromSnapshot();

    void SetLastError(std::string error);

    PhysicsWorldSettings settings_;
    PhysicsSceneSnapshot scene_snapshot_;
    PhysicsSceneState scene_state_;
    std::vector<PhysicsExternalForce> external_forces_;
    std::string last_error_;
};

} // namespace gobot
