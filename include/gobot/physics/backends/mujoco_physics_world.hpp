/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef GOBOT_HAS_MUJOCO
#include <cstddef>
#include <string>
#include <vector>
#endif

#include "gobot/physics/physics_world.hpp"
#include "gobot/physics/joint_controller.hpp"

namespace gobot {

class GOBOT_EXPORT MuJoCoPhysicsWorld : public PhysicsWorld {
    GOBCLASS(MuJoCoPhysicsWorld, PhysicsWorld)

public:
    MuJoCoPhysicsWorld();

    ~MuJoCoPhysicsWorld() override;

    static bool IsBackendAvailable();

    static std::string GetUnavailableReason();

    PhysicsBackendType GetBackendType() const override;

    bool IsAvailable() const override;

    const std::string& GetLastError() const override;

    bool BuildFromScene(const Node* scene_root) override;

    bool RestoreCompatibleState(const PhysicsSceneState& previous_state) override;

    void Reset() override;

    void Step(RealType delta_time) override;

    bool SetLinkExternalForce(const std::string& robot_name,
                              const std::string& link_name,
                              const Vector3& point,
                              const Vector3& force) override;

    bool SetLinkSpringForce(const std::string& robot_name,
                            const std::string& link_name,
                            const Vector3& local_point,
                            const Vector3& target_point,
                            const Vector3& force_hint) override;

    void ClearExternalForces() override;

private:
#ifdef GOBOT_HAS_MUJOCO
    bool LoadModelFromRobotSources();

    bool AttachRobotModelToSpec(void* parent_spec,
                                const PhysicsRobotSnapshot& robot,
                                std::size_t robot_index,
                                const std::string& prefix);

    bool AddAuthoredRobotToSpec(void* parent_spec,
                                const PhysicsRobotSnapshot& robot,
                                std::size_t robot_index,
                                const std::string& prefix);

    void AddLooseSceneGeomsToSpec(void* spec);

    void AddFloatingBaseJointsToSpec(void* spec, const PhysicsRobotSnapshot& robot);

    void BuildJointBindings();

    void BuildLinkBindings();

    std::string GetRobotPrefix(std::size_t robot_index) const;

    void ApplyControlsToMuJoCo();

    void ApplyExternalForcesToMuJoCo();

    void FreeModel();

    void SyncStateFromMuJoCo();

    void SyncStateToMuJoCo();

    void SyncContactsFromMuJoCo();

    struct MuJoCoJointBinding {
        std::size_t robot_index{0};
        std::size_t joint_index{0};
        int mujoco_joint_id{-1};
        int motor_actuator_id{-1};
        int position_actuator_id{-1};
        int velocity_actuator_id{-1};
        int qpos_address{-1};
        int dof_address{-1};
        int joint_type{-1};
        JointController controller;
    };

    struct MuJoCoLinkBinding {
        std::size_t robot_index{0};
        std::size_t link_index{0};
        int body_id{-1};
    };

    struct MuJoCoRobotBinding {
        std::size_t robot_index{0};
        std::string prefix;
    };
#endif

    bool available_{false};

#ifdef GOBOT_HAS_MUJOCO
    void* model_{nullptr};
    void* data_{nullptr};
    std::vector<MuJoCoRobotBinding> robot_bindings_;
    std::vector<MuJoCoJointBinding> joint_bindings_;
    std::vector<MuJoCoLinkBinding> link_bindings_;
#endif
};

} // namespace gobot
