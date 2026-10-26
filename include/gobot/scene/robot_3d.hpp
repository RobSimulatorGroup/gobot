/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <string>
#include <vector>

#include "gobot/scene/node_3d.hpp"

namespace gobot {

enum class RobotMode {
    Assembly,
    Motion
};

class GOBOT_EXPORT Robot3D : public Node3D {
    GOBCLASS(Robot3D, Node3D)

public:
    Robot3D() = default;

    void SetSourcePath(std::string source_path);

    const std::string& GetSourcePath() const;

    // Named link pairs whose geometries are allowed to overlap within the robot.
    void SetCollisionExclusions(std::vector<std::array<std::string, 2>> exclusions);

    const std::vector<std::array<std::string, 2>>& GetCollisionExclusions() const;

    void SetMode(RobotMode mode);

    RobotMode GetMode() const;

private:
    void SetJointMotionModeRecursive(Node* node, bool enabled);

    std::string source_path_;
    std::vector<std::array<std::string, 2>> collision_exclusions_;
    RobotMode mode_{RobotMode::Assembly};
};

}
