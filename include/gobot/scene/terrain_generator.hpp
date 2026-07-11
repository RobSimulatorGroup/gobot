/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include "gobot/scene/resources/terrain_generator_config.hpp"
#include "gobot/scene/terrain_3d.hpp"

namespace gobot {

struct GOBOT_EXPORT GeneratedTerrainData {
    std::vector<TerrainBox> boxes;
    std::vector<TerrainHeightField> heightfields;
    std::vector<TerrainMeshPatch> mesh_patches;
    std::vector<Vector3> spawn_origins;
};

GOBOT_EXPORT bool GenerateTerrain(const TerrainGeneratorConfig& config,
                                  GeneratedTerrainData* result,
                                  std::string* error = nullptr);

} // namespace gobot
