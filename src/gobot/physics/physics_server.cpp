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
        case PhysicsBackendType::PhysXCpu:
            return {
                    PhysicsBackendType::PhysXCpu,
                    "PhysX CPU",
                    false,
                    true,
                    false,
                    false,
                    "PhysX backend is reserved but not implemented yet."
            };
        case PhysicsBackendType::PhysXGpu:
            return {
                    PhysicsBackendType::PhysXGpu,
                    "PhysX GPU",
                    false,
                    false,
                    true,
                    false,
                    "PhysX GPU backend is reserved but not implemented yet."
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
    }

    return {};
}

std::vector<PhysicsBackendInfo> PhysicsServer::GetBackendInfos() const {
    return {
            GetBackendInfo(PhysicsBackendType::Null),
            GetBackendInfo(PhysicsBackendType::MuJoCoCpu),
            GetBackendInfo(PhysicsBackendType::PhysXCpu),
            GetBackendInfo(PhysicsBackendType::PhysXGpu),
            GetBackendInfo(PhysicsBackendType::NewtonGpu),
    };
}

Ref<PhysicsWorld> PhysicsServer::CreateWorld(const PhysicsWorldSettings& settings) {
    return CreateWorldForBackend(backend_type_, settings);
}

Ref<PhysicsWorld> PhysicsServer::CreateWorldForBackend(PhysicsBackendType backend_type,
                                                       const PhysicsWorldSettings& settings) const {
    Ref<PhysicsWorld> world;
    switch (backend_type) {
        case PhysicsBackendType::MuJoCoCpu:
            world = MakeRef<MuJoCoPhysicsWorld>();
            break;
        case PhysicsBackendType::Null:
        case PhysicsBackendType::PhysXCpu:
        case PhysicsBackendType::PhysXGpu:
        case PhysicsBackendType::NewtonGpu:
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
