/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/physics_server.hpp"

#include <mutex>
#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {
namespace {

struct BackendRegistration {
    PhysicsBackendInfo info;
    PhysicsServer::WorldFactory factory;
    PhysicsServer::ArtifactCompiler artifact_compiler;
};

std::vector<BackendRegistration>& BackendRegistry() {
    static std::vector<BackendRegistration> registry;
    return registry;
}

std::mutex& BackendRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

PhysicsBackendInfo MissingBackendInfo(PhysicsBackendType backend_type) {
    switch (backend_type) {
        case PhysicsBackendType::Null:
            return {
                    PhysicsBackendType::Null,
                    "Null",
                    false,
                    true,
                    false,
                    false,
                    "Null physics backend failed to register."
            };
        case PhysicsBackendType::MuJoCoCpu:
            return {
                    PhysicsBackendType::MuJoCoCpu,
                    "MuJoCo CPU",
                    false,
                    true,
                    false,
                    true,
                    "MuJoCo CPU support is not compiled into this build."
            };
    }
    return {};
}

} // namespace

PhysicsServer* PhysicsServer::s_singleton = nullptr;

PhysicsServer::PhysicsServer(PhysicsBackendType backend_type, bool register_singleton)
    : backend_type_(backend_type),
      registered_singleton_(register_singleton) {
    if (registered_singleton_) {
        s_singleton = this;
    }
}

PhysicsServer::~PhysicsServer() {
    if (s_singleton == this) {
        s_singleton = nullptr;
    }
}

PhysicsServer* PhysicsServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initializing PhysicsServer.");
    return s_singleton;
}

bool PhysicsServer::HasInstance() {
    return s_singleton != nullptr;
}

PhysicsBackendType PhysicsServer::GetBackendType() const {
    return backend_type_;
}

void PhysicsServer::SetBackendType(PhysicsBackendType backend_type) {
    backend_type_ = backend_type;
}

bool PhysicsServer::IsBackendAvailable(PhysicsBackendType backend_type) const {
    return GetBackendInfo(backend_type).available;
}

PhysicsBackendInfo PhysicsServer::GetBackendInfo(PhysicsBackendType backend_type) const {
    return GetBackendInfoForBackend(backend_type);
}

PhysicsBackendInfo PhysicsServer::GetBackendInfoForBackend(PhysicsBackendType backend_type) {
    std::scoped_lock lock(BackendRegistryMutex());
    for (const BackendRegistration& registration : BackendRegistry()) {
        if (registration.info.type == backend_type) {
            return registration.info;
        }
    }
    return MissingBackendInfo(backend_type);
}

std::vector<PhysicsBackendInfo> PhysicsServer::GetBackendInfos() const {
    return GetBackendInfosForAllBackends();
}

std::vector<PhysicsBackendInfo> PhysicsServer::GetBackendInfosForAllBackends() {
    return {
            GetBackendInfoForBackend(PhysicsBackendType::Null),
            GetBackendInfoForBackend(PhysicsBackendType::MuJoCoCpu),
    };
}

Ref<PhysicsWorld> PhysicsServer::CreateWorld(const PhysicsWorldSettings& settings) {
    return CreateWorldForBackend(backend_type_, settings);
}

Ref<PhysicsWorld> PhysicsServer::CreateWorldForBackend(PhysicsBackendType backend_type,
                                                       const PhysicsWorldSettings& settings) {
    WorldFactory factory;
    {
        std::scoped_lock lock(BackendRegistryMutex());
        for (const BackendRegistration& registration : BackendRegistry()) {
            if (registration.info.type == backend_type && registration.info.available) {
                factory = registration.factory;
                break;
            }
        }
    }

    if (!factory) {
        return {};
    }
    Ref<PhysicsWorld> world = factory();
    if (!world.IsValid()) {
        return {};
    }
    world->SetSettings(settings);
    return world;
}

bool PhysicsServer::CompileSceneArtifactForBackend(PhysicsBackendType backend_type,
                                                   PhysicsSceneSnapshot scene_snapshot,
                                                   const PhysicsWorldSettings& settings,
                                                   PhysicsSceneArtifact* artifact,
                                                   std::string* error) {
    ArtifactCompiler compiler;
    {
        std::scoped_lock lock(BackendRegistryMutex());
        for (const BackendRegistration& registration : BackendRegistry()) {
            if (registration.info.type == backend_type && registration.info.available) {
                compiler = registration.artifact_compiler;
                break;
            }
        }
    }
    if (!compiler) {
        if (error != nullptr) {
            *error = "Selected physics backend does not provide a scene artifact compiler.";
        }
        return false;
    }
    return compiler(std::move(scene_snapshot), settings, artifact, error);
}

bool PhysicsServer::RegisterBackend(PhysicsBackendInfo info,
                                    WorldFactory factory,
                                    ArtifactCompiler artifact_compiler) {
    if (!factory) {
        return false;
    }

    std::scoped_lock lock(BackendRegistryMutex());
    for (BackendRegistration& registration : BackendRegistry()) {
        if (registration.info.type == info.type) {
            registration = {
                    std::move(info),
                    std::move(factory),
                    std::move(artifact_compiler)};
            return true;
        }
    }
    BackendRegistry().push_back({
            std::move(info),
            std::move(factory),
            std::move(artifact_compiler)});
    return true;
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<PhysicsServer>("PhysicsServer")
            .constructor()(CtorAsRawPtr)
            .property("backend_type", &PhysicsServer::GetBackendType, &PhysicsServer::SetBackendType)
            .method("is_backend_available", &PhysicsServer::IsBackendAvailable)
            .method("create_world", &PhysicsServer::CreateWorld);

};
