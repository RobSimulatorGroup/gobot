/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-27
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace gobot {

enum class NotificationType {
    PostNew,
    PreDelete,

    EnterTree,
    ExitTree,
    PathRenamed,
    Ready,
    Parented,
    Unparented,
    MovedInParent,

    EnterWorld,
    ExitWorld,
    TransformChanged,
    LocalTransformChanged,

    PhysicsProcess,
    Process,

};

}
