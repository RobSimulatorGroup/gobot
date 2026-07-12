/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258551999@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "gobot/physics/physics_types.hpp"

namespace gobot {

class GOBOT_EXPORT MuJoCoSceneCompiler {
public:
    static constexpr std::uint32_t kArtifactSchemaVersion = 1;

    static bool Compile(PhysicsSceneSnapshot scene_snapshot,
                        const PhysicsWorldSettings& settings,
                        PhysicsSceneArtifact* artifact,
                        std::string* error = nullptr);
};

} // namespace gobot
