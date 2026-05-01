/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/physics/physics_world.hpp"

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

    void Reset() override;

    void Step(RealType delta_time) override;

private:
    bool available_{false};
};

} // namespace gobot
