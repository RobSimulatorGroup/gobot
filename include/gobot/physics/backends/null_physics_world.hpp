/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/physics/physics_world.hpp"

namespace gobot {

class GOBOT_EXPORT NullPhysicsWorld : public PhysicsWorld {
    GOBCLASS(NullPhysicsWorld, PhysicsWorld)

public:
    PhysicsBackendType GetBackendType() const override;

    bool IsAvailable() const override;

    const std::string& GetLastError() const override;
};

} // namespace gobot
