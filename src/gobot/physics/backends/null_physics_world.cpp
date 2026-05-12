/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/backends/null_physics_world.hpp"

#include "gobot/core/registration.hpp"

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

} // namespace gobot

GOBOT_REGISTRATION {

    Class_<NullPhysicsWorld>("NullPhysicsWorld")
            .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<NullPhysicsWorld>, Ref<PhysicsWorld>>();

};
