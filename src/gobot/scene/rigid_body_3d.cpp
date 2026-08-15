/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/rigid_body_3d.hpp"

#include "gobot/core/registration.hpp"

GOBOT_REGISTRATION {
    Class_<gobot::RigidBody3D>("RigidBody3D")
            .constructor()(CtorAsRawPtr);
};
