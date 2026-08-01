/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_types.hpp"
#include "gobot/physics/physics_world.hpp"

namespace gobot {

class GOBOT_EXPORT PhysicsServer : public Object {
    GOBCLASS(PhysicsServer, Object)

public:
    using WorldFactory = std::function<Ref<PhysicsWorld>()>;
    using ArtifactCompiler = std::function<bool(PhysicsSceneSnapshot,
                                                const PhysicsWorldSettings&,
                                                PhysicsSceneArtifact*,
                                                std::string*)>;

    explicit PhysicsServer(bool register_singleton = true);

    ~PhysicsServer() override;

    static PhysicsServer* GetInstance();

    static bool HasInstance();

    static bool IsBackendAvailable(PhysicsBackendType backend_type);

    static PhysicsBackendInfo GetBackendInfo(PhysicsBackendType backend_type);

    static std::vector<PhysicsBackendInfo> GetBackendInfos();

    static Ref<PhysicsWorld> CreateWorld(PhysicsBackendType backend_type = PhysicsBackendType::Null,
                                         const PhysicsWorldSettings& settings = {});

    static bool CompileSceneArtifact(PhysicsBackendType backend_type,
                                     PhysicsSceneSnapshot scene_snapshot,
                                     const PhysicsWorldSettings& settings,
                                     PhysicsSceneArtifact* artifact,
                                     std::string* error = nullptr);

    static bool RegisterBackend(PhysicsBackendInfo info,
                                WorldFactory factory,
                                ArtifactCompiler artifact_compiler = {});

private:
    static PhysicsServer* s_singleton;

    bool registered_singleton_{false};
};

} // namespace gobot
