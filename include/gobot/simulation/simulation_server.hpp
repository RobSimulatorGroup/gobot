/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/physics/physics_world.hpp"

namespace gobot {

class Node;

class GOBOT_EXPORT SimulationServer : public Object {
    GOBCLASS(SimulationServer, Object)

public:
    explicit SimulationServer(PhysicsBackendType backend_type = PhysicsBackendType::Null);

    ~SimulationServer() override;

    static SimulationServer* GetInstance();

    static bool HasInstance();

    PhysicsBackendType GetBackendType() const;

    void SetBackendType(PhysicsBackendType backend_type);

    const PhysicsWorldSettings& GetPhysicsWorldSettings() const;

    void SetPhysicsWorldSettings(const PhysicsWorldSettings& settings);

    const JointControllerGains& GetDefaultJointGains() const;

    void SetDefaultJointGains(const JointControllerGains& gains);

    RealType GetFixedTimeStep() const;

    void SetFixedTimeStep(RealType fixed_time_step);

    RealType GetTimeScale() const;

    void SetTimeScale(RealType time_scale);

    int GetMaxSubSteps() const;

    void SetMaxSubSteps(int max_sub_steps);

    bool IsPaused() const;

    void SetPaused(bool paused);

    bool BuildWorldFromScene(const Node* scene_root);

    bool RebuildWorldFromScene(const Node* scene_root, bool preserve_state = true);

    const Node* GetSceneRoot() const;

    void ClearWorld();

    bool HasWorld() const;

    Ref<PhysicsWorld> GetWorld() const;

    bool Reset();

    bool StepOnce();

    int Step(RealType delta_time);

    bool SyncSceneFromWorld();

    bool SetJointPositionTarget(const std::string& robot_name,
                                const std::string& joint_name,
                                RealType target_position);

    bool SetJointVelocityTarget(const std::string& robot_name,
                                const std::string& joint_name,
                                RealType target_velocity);

    bool SetJointEffortTarget(const std::string& robot_name,
                              const std::string& joint_name,
                              RealType target_effort);

    bool SetJointPassive(const std::string& robot_name,
                         const std::string& joint_name);

    bool ResetJointState(const std::string& robot_name,
                         const std::string& joint_name,
                         RealType position,
                         RealType velocity);

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

    RealType GetSimulationTime() const;

    std::uint64_t GetFrameCount() const;

    RealType GetAccumulator() const;

    int GetLastStepCount() const;

    const std::string& GetLastError() const;

private:
    bool EnsureWorldReady();

    bool StepFixed();

    bool ApplyWorldStateToScene();

    void ResetClock();

    void SetLastError(std::string error);

    static SimulationServer* s_singleton;

    PhysicsBackendType backend_type_{PhysicsBackendType::Null};
    PhysicsWorldSettings physics_world_settings_;
    Ref<PhysicsWorld> world_;
    const Node* scene_root_{nullptr};
    bool paused_{true};
    RealType time_scale_{1.0};
    int max_sub_steps_{8};
    int last_step_count_{0};
    RealType accumulator_{0.0};
    RealType simulation_time_{0.0};
    std::uint64_t frame_count_{0};
    std::string last_error_;
};

} // namespace gobot
