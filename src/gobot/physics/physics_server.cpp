/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/physics/physics_server.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/physics/backends/mujoco_physics_world.hpp"
#include "gobot/physics/backends/null_physics_world.hpp"

namespace gobot {

PhysicsServer* PhysicsServer::s_singleton = nullptr;

PhysicsServer::PhysicsServer(PhysicsBackendType backend_type)
    : backend_type_(backend_type) {
    s_singleton = this;
}

PhysicsServer::~PhysicsServer() {
    s_singleton = nullptr;
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
    switch (backend_type) {
        case PhysicsBackendType::Null:
            return {
                    PhysicsBackendType::Null,
                    "Null",
                    true,
                    true,
                    false,
                    false,
                    "No-op physics backend for editor and tests."
            };
        case PhysicsBackendType::MuJoCoCpu:
            return {
                    PhysicsBackendType::MuJoCoCpu,
                    "MuJoCo CPU",
                    MuJoCoPhysicsWorld::IsBackendAvailable(),
                    true,
                    false,
                    true,
                    MuJoCoPhysicsWorld::IsBackendAvailable()
                            ? "Available."
                            : MuJoCoPhysicsWorld::GetUnavailableReason()
            };
        case PhysicsBackendType::MuJoCoWarp:
            return {
                    PhysicsBackendType::MuJoCoWarp,
                    "MuJoCo Warp",
                    false,
                    false,
                    true,
                    true,
                    "MuJoCo Warp backend is planned for CUDA graph vector simulation, but is not implemented yet."
            };
        case PhysicsBackendType::NewtonGpu:
            return {
                    PhysicsBackendType::NewtonGpu,
                    "Newton GPU",
                    false,
                    false,
                    true,
                    true,
                    "Newton backend is reserved but not implemented yet."
            };
        case PhysicsBackendType::RigidIpcCpu:
            return {
                    PhysicsBackendType::RigidIpcCpu,
                    "Rigid IPC CPU",
                    false,
                    true,
                    false,
                    false,
                    "Rigid IPC backend is reserved for robust intersection-free contact validation, but not implemented yet."
            };
    }

    return {};
}

std::vector<PhysicsBackendInfo> PhysicsServer::GetBackendInfos() const {
    return GetBackendInfosForAllBackends();
}

std::vector<PhysicsBackendInfo> PhysicsServer::GetBackendInfosForAllBackends() {
    return {
            GetBackendInfoForBackend(PhysicsBackendType::Null),
            GetBackendInfoForBackend(PhysicsBackendType::MuJoCoCpu),
            GetBackendInfoForBackend(PhysicsBackendType::MuJoCoWarp),
            GetBackendInfoForBackend(PhysicsBackendType::NewtonGpu),
            GetBackendInfoForBackend(PhysicsBackendType::RigidIpcCpu),
    };
}

Ref<PhysicsWorld> PhysicsServer::CreateWorld(const PhysicsWorldSettings& settings) {
    return CreateWorldForBackend(backend_type_, settings);
}

Ref<PhysicsWorld> PhysicsServer::CreateWorldForBackend(PhysicsBackendType backend_type,
                                                       const PhysicsWorldSettings& settings) {
    Ref<PhysicsWorld> world;
    switch (backend_type) {
        case PhysicsBackendType::MuJoCoCpu:
            world = MakeRef<MuJoCoPhysicsWorld>();
            break;
        case PhysicsBackendType::Null:
        case PhysicsBackendType::MuJoCoWarp:
        case PhysicsBackendType::NewtonGpu:
        case PhysicsBackendType::RigidIpcCpu:
            world = MakeRef<NullPhysicsWorld>();
            break;
    }

    world->SetSettings(settings);
    return world;
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<PhysicsServer>("PhysicsServer")
            .constructor()(CtorAsRawPtr)
            .property("backend_type", &PhysicsServer::GetBackendType, &PhysicsServer::SetBackendType)
            .method("is_backend_available", &PhysicsServer::IsBackendAvailable)
            .method("create_world", &PhysicsServer::CreateWorld);

};
