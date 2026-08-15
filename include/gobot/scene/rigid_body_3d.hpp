/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/link_3d.hpp"

namespace gobot {

// A free six-degree-of-freedom rigid body. It reuses Link3D's inertial
// contract, while remaining a standalone scene object rather than an
// articulation root.
class GOBOT_EXPORT RigidBody3D : public Link3D {
    GOBCLASS(RigidBody3D, Link3D)

public:
    RigidBody3D() = default;
};

} // namespace gobot
