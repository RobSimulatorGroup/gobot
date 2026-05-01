/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
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

    virtual void Reset();

    virtual void Step(RealType delta_time);

    bool SetJointControl(const std::string& robot_name,
                         const std::string& joint_name,
                         PhysicsJointControlMode control_mode,
                         RealType target);

    const PhysicsSceneSnapshot& GetSceneSnapshot() const;

    const PhysicsSceneState& GetSceneState() const;

protected:
    bool CaptureSceneSnapshot(const Node* scene_root);

    PhysicsJointState* FindJointState(const std::string& robot_name,
                                      const std::string& joint_name);

    void ResetSceneStateFromSnapshot();

    void SetLastError(std::string error);

    PhysicsWorldSettings settings_;
    PhysicsSceneSnapshot scene_snapshot_;
    PhysicsSceneState scene_state_;
    std::string last_error_;
};

} // namespace gobot
