/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/geometry.hpp"

namespace gobot {

enum class TerrainSubTerrainType {
    Flat,
    PyramidStairs,
    PyramidSlope,
    RandomRough,
    Wave,
};

struct GOBOT_EXPORT TerrainSubTerrainConfig {
    std::string name;
    TerrainSubTerrainType type{TerrainSubTerrainType::Flat};
    RealType proportion{1.0};
    bool inverted{false};
    Vector2 step_height_range{0.0, 0.1};
    RealType step_width{0.3};
    RealType platform_width{3.0};
    Vector2 slope_range{0.0, 1.0};
    Vector2 noise_range{0.02, 0.1};
    RealType noise_step{0.02};
    RealType downsampled_scale{0.0};
    Vector2 amplitude_range{0.0, 0.2};
    RealType num_waves{4.0};
    RealType border_width{0.0};
};

class GOBOT_EXPORT TerrainGeneratorConfig : public Resource {
    GOBCLASS(TerrainGeneratorConfig, Resource)

public:
    static constexpr int kCurrentSchemaVersion = 1;

    TerrainGeneratorConfig() = default;

    int GetSchemaVersion() const;
    void SetSchemaVersion(int schema_version);

    std::uint64_t GetSeed() const;
    void SetSeed(std::uint64_t seed);

    bool IsCurriculum() const;
    void SetCurriculum(bool curriculum);

    int GetNumRows() const;
    void SetNumRows(int num_rows);

    int GetNumCols() const;
    void SetNumCols(int num_cols);

    const Vector2& GetPatchSize() const;
    void SetPatchSize(const Vector2& patch_size);

    RealType GetBorderWidth() const;
    void SetBorderWidth(RealType border_width);

    const Vector2& GetDifficultyRange() const;
    void SetDifficultyRange(const Vector2& difficulty_range);

    RealType GetHorizontalScale() const;
    void SetHorizontalScale(RealType horizontal_scale);

    const std::vector<TerrainSubTerrainConfig>& GetSubTerrains() const;
    void SetSubTerrains(const std::vector<TerrainSubTerrainConfig>& sub_terrains);

    std::size_t GetContentHash() const;

private:
    int schema_version_{kCurrentSchemaVersion};
    std::uint64_t seed_{42};
    bool curriculum_{true};
    int num_rows_{10};
    int num_cols_{20};
    Vector2 patch_size_{8.0, 8.0};
    RealType border_width_{20.0};
    Vector2 difficulty_range_{0.0, 1.0};
    RealType horizontal_scale_{0.1};
    std::vector<TerrainSubTerrainConfig> sub_terrains_;
};

GOBOT_EXPORT Ref<TerrainGeneratorConfig> MakeRoughTerrainGeneratorConfig();

} // namespace gobot
