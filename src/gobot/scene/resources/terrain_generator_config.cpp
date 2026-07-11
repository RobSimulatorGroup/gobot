/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/terrain_generator_config.hpp"

#include <algorithm>

#include "gobot/core/hash_combine.hpp"
#include "gobot/core/registration.hpp"

namespace gobot {

int TerrainGeneratorConfig::GetSchemaVersion() const {
    return schema_version_;
}

void TerrainGeneratorConfig::SetSchemaVersion(int schema_version) {
    schema_version_ = schema_version;
}

std::uint64_t TerrainGeneratorConfig::GetSeed() const {
    return seed_;
}

void TerrainGeneratorConfig::SetSeed(std::uint64_t seed) {
    seed_ = seed;
}

bool TerrainGeneratorConfig::IsCurriculum() const {
    return curriculum_;
}

void TerrainGeneratorConfig::SetCurriculum(bool curriculum) {
    curriculum_ = curriculum;
}

int TerrainGeneratorConfig::GetNumRows() const {
    return num_rows_;
}

void TerrainGeneratorConfig::SetNumRows(int num_rows) {
    num_rows_ = num_rows;
}

int TerrainGeneratorConfig::GetNumCols() const {
    return num_cols_;
}

void TerrainGeneratorConfig::SetNumCols(int num_cols) {
    num_cols_ = num_cols;
}

const Vector2& TerrainGeneratorConfig::GetPatchSize() const {
    return patch_size_;
}

void TerrainGeneratorConfig::SetPatchSize(const Vector2& patch_size) {
    patch_size_ = patch_size;
}

RealType TerrainGeneratorConfig::GetBorderWidth() const {
    return border_width_;
}

void TerrainGeneratorConfig::SetBorderWidth(RealType border_width) {
    border_width_ = border_width;
}

const Vector2& TerrainGeneratorConfig::GetDifficultyRange() const {
    return difficulty_range_;
}

void TerrainGeneratorConfig::SetDifficultyRange(const Vector2& difficulty_range) {
    difficulty_range_ = difficulty_range;
}

RealType TerrainGeneratorConfig::GetHorizontalScale() const {
    return horizontal_scale_;
}

void TerrainGeneratorConfig::SetHorizontalScale(RealType horizontal_scale) {
    horizontal_scale_ = horizontal_scale;
}

const std::vector<TerrainSubTerrainConfig>& TerrainGeneratorConfig::GetSubTerrains() const {
    return sub_terrains_;
}

void TerrainGeneratorConfig::SetSubTerrains(const std::vector<TerrainSubTerrainConfig>& sub_terrains) {
    sub_terrains_ = sub_terrains;
}

std::size_t TerrainGeneratorConfig::GetContentHash() const {
    std::size_t hash = 0;
    HashCombine(hash,
                schema_version_,
                seed_,
                curriculum_,
                num_rows_,
                num_cols_,
                patch_size_.x(),
                patch_size_.y(),
                border_width_,
                difficulty_range_.x(),
                difficulty_range_.y(),
                horizontal_scale_,
                sub_terrains_.size());
    for (const TerrainSubTerrainConfig& terrain : sub_terrains_) {
        HashCombine(hash,
                    terrain.name,
                    static_cast<int>(terrain.type),
                    terrain.proportion,
                    terrain.inverted,
                    terrain.step_height_range.x(),
                    terrain.step_height_range.y(),
                    terrain.step_width,
                    terrain.platform_width,
                    terrain.slope_range.x(),
                    terrain.slope_range.y(),
                    terrain.noise_range.x(),
                    terrain.noise_range.y(),
                    terrain.noise_step,
                    terrain.downsampled_scale,
                    terrain.amplitude_range.x(),
                    terrain.amplitude_range.y(),
                    terrain.num_waves,
                    terrain.border_width);
    }
    return hash;
}

Ref<TerrainGeneratorConfig> MakeRoughTerrainGeneratorConfig() {
    Ref<TerrainGeneratorConfig> config = MakeRef<TerrainGeneratorConfig>();
    config->SetName("Rough Terrain");
    config->SetSchemaVersion(TerrainGeneratorConfig::kCurrentSchemaVersion);
    config->SetSeed(42);
    config->SetCurriculum(true);
    config->SetNumRows(10);
    config->SetNumCols(20);
    config->SetPatchSize({8.0, 8.0});
    config->SetBorderWidth(20.0);
    config->SetDifficultyRange({0.0, 1.0});
    config->SetHorizontalScale(0.1);

    TerrainSubTerrainConfig flat;
    flat.name = "flat";
    flat.type = TerrainSubTerrainType::Flat;
    flat.proportion = 0.2;

    TerrainSubTerrainConfig stairs;
    stairs.name = "pyramid_stairs";
    stairs.type = TerrainSubTerrainType::PyramidStairs;
    stairs.proportion = 0.2;
    stairs.step_height_range = {0.0, 0.1};
    stairs.step_width = 0.3;
    stairs.platform_width = 3.0;
    stairs.border_width = 1.0;

    TerrainSubTerrainConfig stairs_inverted = stairs;
    stairs_inverted.name = "pyramid_stairs_inverted";
    stairs_inverted.inverted = true;

    TerrainSubTerrainConfig slope;
    slope.name = "pyramid_slope";
    slope.type = TerrainSubTerrainType::PyramidSlope;
    slope.proportion = 0.1;
    slope.slope_range = {0.0, 1.0};
    slope.platform_width = 2.0;
    slope.border_width = 0.25;

    TerrainSubTerrainConfig slope_inverted = slope;
    slope_inverted.name = "pyramid_slope_inverted";
    slope_inverted.inverted = true;

    TerrainSubTerrainConfig rough;
    rough.name = "random_rough";
    rough.type = TerrainSubTerrainType::RandomRough;
    rough.proportion = 0.1;
    rough.noise_range = {0.02, 0.1};
    rough.noise_step = 0.02;
    rough.downsampled_scale = 0.0;
    rough.border_width = 0.25;

    TerrainSubTerrainConfig wave;
    wave.name = "wave";
    wave.type = TerrainSubTerrainType::Wave;
    wave.proportion = 0.1;
    wave.amplitude_range = {0.0, 0.2};
    wave.num_waves = 4.0;
    wave.border_width = 0.25;

    config->SetSubTerrains({
            flat,
            stairs,
            stairs_inverted,
            slope,
            slope_inverted,
            rough,
            wave,
    });
    return config;
}

} // namespace gobot

GOBOT_REGISTRATION {
    gobot::QuickEnumeration_<gobot::TerrainSubTerrainType>("TerrainSubTerrainType");

    Class_<gobot::TerrainSubTerrainConfig>("TerrainSubTerrainConfig")
            .constructor()(CtorAsObject)
            .property("name", &gobot::TerrainSubTerrainConfig::name)
            .property("type", &gobot::TerrainSubTerrainConfig::type)
            .property("proportion", &gobot::TerrainSubTerrainConfig::proportion)
            .property("inverted", &gobot::TerrainSubTerrainConfig::inverted)
            .property("step_height_range", &gobot::TerrainSubTerrainConfig::step_height_range)
            .property("step_width", &gobot::TerrainSubTerrainConfig::step_width)
            .property("platform_width", &gobot::TerrainSubTerrainConfig::platform_width)
            .property("slope_range", &gobot::TerrainSubTerrainConfig::slope_range)
            .property("noise_range", &gobot::TerrainSubTerrainConfig::noise_range)
            .property("noise_step", &gobot::TerrainSubTerrainConfig::noise_step)
            .property("downsampled_scale", &gobot::TerrainSubTerrainConfig::downsampled_scale)
            .property("amplitude_range", &gobot::TerrainSubTerrainConfig::amplitude_range)
            .property("num_waves", &gobot::TerrainSubTerrainConfig::num_waves)
            .property("border_width", &gobot::TerrainSubTerrainConfig::border_width);

    Class_<gobot::TerrainGeneratorConfig>("TerrainGeneratorConfig")
            .constructor()(CtorAsRawPtr)
            .property("schema_version",
                      &gobot::TerrainGeneratorConfig::GetSchemaVersion,
                      &gobot::TerrainGeneratorConfig::SetSchemaVersion)
            .property("seed", &gobot::TerrainGeneratorConfig::GetSeed, &gobot::TerrainGeneratorConfig::SetSeed)
            .property("curriculum",
                      &gobot::TerrainGeneratorConfig::IsCurriculum,
                      &gobot::TerrainGeneratorConfig::SetCurriculum)
            .property("num_rows",
                      &gobot::TerrainGeneratorConfig::GetNumRows,
                      &gobot::TerrainGeneratorConfig::SetNumRows)
            .property("num_cols",
                      &gobot::TerrainGeneratorConfig::GetNumCols,
                      &gobot::TerrainGeneratorConfig::SetNumCols)
            .property("patch_size",
                      &gobot::TerrainGeneratorConfig::GetPatchSize,
                      &gobot::TerrainGeneratorConfig::SetPatchSize)
            .property("border_width",
                      &gobot::TerrainGeneratorConfig::GetBorderWidth,
                      &gobot::TerrainGeneratorConfig::SetBorderWidth)
            .property("difficulty_range",
                      &gobot::TerrainGeneratorConfig::GetDifficultyRange,
                      &gobot::TerrainGeneratorConfig::SetDifficultyRange)
            .property("horizontal_scale",
                      &gobot::TerrainGeneratorConfig::GetHorizontalScale,
                      &gobot::TerrainGeneratorConfig::SetHorizontalScale)
            .property("sub_terrains",
                      &gobot::TerrainGeneratorConfig::GetSubTerrains,
                      &gobot::TerrainGeneratorConfig::SetSubTerrains);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::TerrainGeneratorConfig>, gobot::Ref<gobot::Resource>>();
}
