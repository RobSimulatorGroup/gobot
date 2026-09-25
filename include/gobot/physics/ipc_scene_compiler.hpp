/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/physics/ipc_scene_artifact.hpp"
#include "gobot/physics/physics_types.hpp"

namespace gobot {

class GOBOT_EXPORT IpcSceneCompiler {
public:
    static bool Compile(const PhysicsSceneSnapshot& snapshot,
                        IpcSceneArtifact* artifact,
                        std::string* error = nullptr);
};

} // namespace gobot
