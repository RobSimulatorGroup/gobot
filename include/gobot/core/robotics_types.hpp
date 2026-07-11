/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace gobot {

enum class JointType {
    Fixed,
    Revolute,
    Continuous,
    Prismatic,
    Floating,
    Planar
};

enum class JointDriveMode {
    Passive,
    Motor,
    Position,
    Velocity
};

enum class RayReductionMode {
    None,
    Min,
    Max,
    Mean,
};

enum class RayPatternMode {
    Custom,
    Grid,
};

enum class RayAlignmentMode {
    World,
    Base,
    Yaw,
};

} // namespace gobot
