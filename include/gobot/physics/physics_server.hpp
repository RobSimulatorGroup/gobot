/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <memory>

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_types.hpp"
#include "gobot/physics/physics_world.hpp"

namespace gobot {

class GOBOT_EXPORT PhysicsServer : public Object {
    GOBCLASS(PhysicsServer, Object)

public:
    explicit PhysicsServer(PhysicsBackendType backend_type = PhysicsBackendType::Null);

    ~PhysicsServer() override;

    static PhysicsServer* GetInstance();

    static bool HasInstance();

    PhysicsBackendType GetBackendType() const;

    void SetBackendType(PhysicsBackendType backend_type);

    bool IsBackendAvailable(PhysicsBackendType backend_type) const;

    PhysicsBackendInfo GetBackendInfo(PhysicsBackendType backend_type) const;

    std::vector<PhysicsBackendInfo> GetBackendInfos() const;

    Ref<PhysicsWorld> CreateWorld(const PhysicsWorldSettings& settings = {});

private:
    Ref<PhysicsWorld> CreateWorldForBackend(PhysicsBackendType backend_type,
                                            const PhysicsWorldSettings& settings) const;

    static PhysicsServer* s_singleton;

    PhysicsBackendType backend_type_{PhysicsBackendType::Null};
};

} // namespace gobot
