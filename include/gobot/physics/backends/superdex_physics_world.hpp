/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "gobot/physics/physics_world.hpp"

namespace gobot {

class GOBOT_EXPORT SuperDexPhysicsWorld : public PhysicsWorld {
    GOBCLASS(SuperDexPhysicsWorld, PhysicsWorld)

public:
    SuperDexPhysicsWorld();

    ~SuperDexPhysicsWorld() override;

    PhysicsBackendType GetBackendType() const override;

    bool IsAvailable() const override;

    const std::string& GetLastError() const override;

    PhysicsBackendCapabilities GetCapabilities() const override;

    PhysicsSolverDiagnostics GetSolverDiagnostics() const override;

    bool Build(PhysicsSceneSnapshot scene_snapshot) override;

    bool RestoreCompatibleState(const PhysicsSceneState& previous_state) override;

    void Reset() override;

    PhysicsStepResult Step(RealType delta_time) override;

    Ref<PhysicsRuntimeCheckpoint> CaptureCheckpoint() const override;

    bool RestoreCheckpoint(
            const Ref<PhysicsRuntimeCheckpoint>& checkpoint,
            const std::vector<std::size_t>& environment_indices = {}) override;

    bool ResetJointState(const std::string& robot_name,
                         const std::string& joint_name,
                         RealType position,
                         RealType velocity = 0.0) override;

    bool ResetLinkState(const std::string& robot_name,
                        const std::string& link_name,
                        const Vector3& position,
                        const Quaternion& orientation = Quaternion::Identity(),
                        const Vector3& linear_velocity = Vector3::Zero(),
                        const Vector3& angular_velocity = Vector3::Zero()) override;

    bool WriteEnvironmentLinkVelocity(std::size_t environment_index,
                                      const std::string& robot_name,
                                      const std::string& link_name,
                                      const Vector3& linear_velocity,
                                      const Vector3& angular_velocity) override;

    PhysicsRaycastHit RaycastTerrain(const PhysicsRaycastQuery& query) const override;

private:
    bool ApplyForces(RealType delta_time);

    bool PushStateToSuperDex();

    bool SyncStateFromSuperDex(RealType delta_time);

    void UpdateDiagnostics();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gobot
