/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/backends/null_physics_world.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

PhysicsBackendType NullPhysicsWorld::GetBackendType() const {
    return PhysicsBackendType::Null;
}

bool NullPhysicsWorld::IsAvailable() const {
    return true;
}

const std::string& NullPhysicsWorld::GetLastError() const {
    return last_error_;
}

PhysicsRaycastHit NullPhysicsWorld::RaycastTerrain(const PhysicsRaycastQuery& query) const {
    if (!reported_missing_native_raycast_.exchange(true)) {
        LOG_WARN("Null physics backend has no native terrain raycast; using Gobot geometry fallback.");
    }
    return RaycastTerrainFallback(query);
}

PhysicsRaycastHit NullPhysicsWorld::RaycastTerrainForSensor(const PhysicsRaycastQuery& query,
                                                            std::size_t environment_index) const {
    GOB_UNUSED(environment_index);
    return RaycastTerrain(query);
}

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<NullPhysicsWorld>("NullPhysicsWorld")
            .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<NullPhysicsWorld>, Ref<PhysicsWorld>>();

};
