/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/terrain_generator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>

namespace gobot {
namespace {

constexpr RealType kHeightfieldVerticalScale = 0.005;
const Color kBlue{0.25f, 0.35f, 0.55f, 1.0f};
const Color kRed{0.65f, 0.28f, 0.22f, 1.0f};
const Color kPlatform{0.44f, 0.46f, 0.42f, 1.0f};
const Color kFlat{0.5f, 0.5f, 0.5f, 1.0f};

class StableRng {
public:
    explicit StableRng(std::uint64_t seed) : state_(seed) {}

    std::uint64_t Next() {
        std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    RealType Uniform() {
        constexpr RealType inverse = 1.0 / static_cast<RealType>(std::uint64_t{1} << 53U);
        return static_cast<RealType>(Next() >> 11U) * inverse;
    }

    RealType Uniform(RealType lower, RealType upper) {
        return lower + (upper - lower) * Uniform();
    }

    std::size_t Index(std::size_t count) {
        return count == 0 ? 0 : static_cast<std::size_t>(Next() % count);
    }

private:
    std::uint64_t state_;
};

Color Darken(const Color& color, RealType amount) {
    return {
            static_cast<float>(std::clamp<RealType>(color.red() * amount, 0.0, 1.0)),
            static_cast<float>(std::clamp<RealType>(color.green() * amount, 0.0, 1.0)),
            static_cast<float>(std::clamp<RealType>(color.blue() * amount, 0.0, 1.0)),
            color.alpha(),
    };
}

void AddBox(GeneratedTerrainData* result,
            const Vector3& center,
            const Vector3& size,
            const Color& color) {
    TerrainBox box;
    box.center = center;
    box.size = size.cwiseMax(Vector3::Constant(2.0e-6));
    box.color = color;
    result->boxes.push_back(std::move(box));
}

void AddBorderBoxes(GeneratedTerrainData* result,
                    const Vector2& size,
                    const Vector2& inner_size,
                    RealType height,
                    const Vector3& position,
                    const Color& color) {
    const RealType thickness_x = (size.x() - inner_size.x()) * 0.5f;
    const RealType thickness_y = (size.y() - inner_size.y()) * 0.5f;
    if (thickness_y > 0.0) {
        AddBox(result,
               {position.x(), position.y() + inner_size.y() * 0.5f + thickness_y * 0.5f, position.z()},
               {size.x(), thickness_y, height},
               color);
        AddBox(result,
               {position.x(), position.y() - inner_size.y() * 0.5f - thickness_y * 0.5f, position.z()},
               {size.x(), thickness_y, height},
               color);
    }
    if (thickness_x > 0.0) {
        AddBox(result,
               {position.x() - inner_size.x() * 0.5f - thickness_x * 0.5f, position.y(), position.z()},
               {thickness_x, inner_size.y(), height},
               color);
        AddBox(result,
               {position.x() + inner_size.x() * 0.5f + thickness_x * 0.5f, position.y(), position.z()},
               {thickness_x, inner_size.y(), height},
               color);
    }
}

RealType DifficultyValue(const Vector2& range, RealType difficulty) {
    return range.x() + (range.y() - range.x()) * std::clamp<RealType>(difficulty, 0.0, 1.0);
}

Vector3 AddStairs(GeneratedTerrainData* result,
                  const TerrainGeneratorConfig& config,
                  const TerrainSubTerrainConfig& terrain,
                  const Vector3& corner,
                  RealType difficulty) {
    const Vector2 patch_size = config.GetPatchSize();
    const RealType step_height = DifficultyValue(terrain.step_height_range, difficulty);
    const RealType step_width = std::max<RealType>(terrain.step_width, 1.0e-6);
    const RealType patch_border = std::max<RealType>(terrain.border_width, 0.0);
    const Vector2 terrain_size = patch_size - Vector2::Constant(2.0f * patch_border);
    const auto step_count = [step_width](RealType length, RealType platform_width) {
        const RealType count = (length - platform_width) / (2.0f * step_width);
        return static_cast<int>(std::floor(count + 1.0e-6));
    };
    const int steps_x = step_count(terrain_size.x(), terrain.platform_width);
    const int steps_y = step_count(terrain_size.y(), terrain.platform_width);
    const int num_steps = std::max(0, std::min(steps_x, steps_y));
    const Vector3 center = corner + Vector3{patch_size.x() * 0.5f, patch_size.y() * 0.5f, 0.0f};
    const Color color = terrain.inverted ? kRed : kBlue;

    if (patch_border > 0.0) {
        AddBorderBoxes(result,
                       patch_size,
                       terrain_size,
                       std::max<RealType>(step_height, 2.0e-6),
                       {center.x(), center.y(), -0.5f * step_height},
                       Darken(color, 0.85));
    }

    if (terrain.inverted) {
        const RealType total_height = (num_steps + 1) * step_height;
        for (int step = 0; step < num_steps; ++step) {
            const Vector2 box_size = terrain_size - Vector2::Constant(2.0f * step * step_width);
            const RealType box_z = -total_height * 0.5f - (step + 1) * step_height * 0.5f;
            const RealType box_offset = (step + 0.5f) * step_width;
            const RealType box_height = total_height - (step + 1) * step_height;
            AddBox(result, {center.x(), center.y() + terrain_size.y() * 0.5f - box_offset, box_z},
                   {box_size.x(), step_width, box_height}, color);
            AddBox(result, {center.x(), center.y() - terrain_size.y() * 0.5f + box_offset, box_z},
                   {box_size.x(), step_width, box_height}, color);
            const RealType side_length = box_size.y() - 2.0f * step_width;
            AddBox(result, {center.x() + terrain_size.x() * 0.5f - box_offset, center.y(), box_z},
                   {step_width, side_length, box_height}, color);
            AddBox(result, {center.x() - terrain_size.x() * 0.5f + box_offset, center.y(), box_z},
                   {step_width, side_length, box_height}, color);
        }
        AddBox(result,
               {center.x(), center.y(), -total_height - step_height * 0.5f},
               {terrain_size.x() - 2.0f * num_steps * step_width,
                terrain_size.y() - 2.0f * num_steps * step_width,
                step_height},
               color);
        return {center.x(), center.y(), -(num_steps + 1) * step_height};
    }

    for (int step = 0; step < num_steps; ++step) {
        const Vector2 box_size = terrain_size - Vector2::Constant(2.0f * step * step_width);
        const RealType box_z = step * step_height * 0.5f;
        const RealType box_offset = (step + 0.5f) * step_width;
        const RealType box_height = (step + 2) * step_height;
        AddBox(result, {center.x(), center.y() + terrain_size.y() * 0.5f - box_offset, box_z},
               {box_size.x(), step_width, box_height}, color);
        AddBox(result, {center.x(), center.y() - terrain_size.y() * 0.5f + box_offset, box_z},
               {box_size.x(), step_width, box_height}, color);
        const RealType side_length = box_size.y() - 2.0f * step_width;
        AddBox(result, {center.x() + terrain_size.x() * 0.5f - box_offset, center.y(), box_z},
               {step_width, side_length, box_height}, color);
        AddBox(result, {center.x() - terrain_size.x() * 0.5f + box_offset, center.y(), box_z},
               {step_width, side_length, box_height}, color);
    }
    AddBox(result,
           {center.x(), center.y(), num_steps * step_height * 0.5f},
           {terrain_size.x() - 2.0f * num_steps * step_width,
            terrain_size.y() - 2.0f * num_steps * step_width,
            (num_steps + 2) * step_height},
           color);
    return {center.x(), center.y(), (num_steps + 1) * step_height};
}

struct HeightfieldNoise {
    int rows{0};
    int cols{0};
    std::vector<int> values;
    RealType z_offset{0.0};
    RealType spawn_height{0.0};
    RealType base_thickness_ratio{1.0};
};

int NoiseAt(const HeightfieldNoise& noise, int row, int col) {
    return noise.values[static_cast<std::size_t>(row * noise.cols + col)];
}

int& NoiseAt(HeightfieldNoise& noise, int row, int col) {
    return noise.values[static_cast<std::size_t>(row * noise.cols + col)];
}

HeightfieldNoise MakeSlopeNoise(const TerrainGeneratorConfig& config,
                                const TerrainSubTerrainConfig& terrain,
                                RealType difficulty) {
    const RealType horizontal_scale = config.GetHorizontalScale();
    const Vector2 patch_size = config.GetPatchSize();
    HeightfieldNoise noise;
    noise.rows = static_cast<int>(patch_size.x() / horizontal_scale);
    noise.cols = static_cast<int>(patch_size.y() / horizontal_scale);
    noise.values.assign(static_cast<std::size_t>(noise.rows * noise.cols), 0);
    const int border = static_cast<int>(terrain.border_width / horizontal_scale);
    const int inner_rows = border > 0 ? noise.rows - 2 * border : noise.rows;
    const int inner_cols = border > 0 ? noise.cols - 2 * border : noise.cols;
    RealType slope = DifficultyValue(terrain.slope_range, difficulty);
    if (terrain.inverted) {
        slope = -slope;
    }
    const int height_max = static_cast<int>(slope * inner_rows * horizontal_scale * 0.5 /
                                            kHeightfieldVerticalScale);
    const int center_row = inner_rows / 2;
    const int center_col = inner_cols / 2;
    const int platform_half = static_cast<int>(terrain.platform_width / horizontal_scale / 2.0);
    const int platform_row = std::clamp(center_row - platform_half, 0, std::max(inner_rows - 1, 0));
    const int platform_col = std::clamp(center_col - platform_half, 0, std::max(inner_cols - 1, 0));
    const auto raw_height = [=](int row, int col) {
        const RealType row_weight = center_row == 0
                                            ? 0.0
                                            : static_cast<RealType>(center_row - std::abs(center_row - row)) /
                                                      center_row;
        const RealType col_weight = center_col == 0
                                            ? 0.0
                                            : static_cast<RealType>(center_col - std::abs(center_col - col)) /
                                                      center_col;
        return static_cast<RealType>(height_max) * row_weight * col_weight;
    };
    const RealType platform_height = raw_height(platform_row, platform_col);
    const RealType clip_low = std::min<RealType>(0.0, platform_height);
    const RealType clip_high = std::max<RealType>(0.0, platform_height);
    for (int row = 0; row < inner_rows; ++row) {
        for (int col = 0; col < inner_cols; ++col) {
            NoiseAt(noise, row + border, col + border) = static_cast<int>(
                    std::nearbyint(std::clamp(raw_height(row, col), clip_low, clip_high)));
        }
    }
    const auto [minimum, maximum] = std::minmax_element(noise.values.begin(), noise.values.end());
    const RealType max_height = std::max(*maximum - *minimum, 1) * kHeightfieldVerticalScale;
    noise.z_offset = terrain.inverted ? -max_height : 0.0;
    noise.spawn_height = terrain.inverted ? noise.z_offset : max_height;
    return noise;
}

HeightfieldNoise MakeRandomRoughNoise(const TerrainGeneratorConfig& config,
                                     const TerrainSubTerrainConfig& terrain,
                                     StableRng* rng) {
    const RealType horizontal_scale = config.GetHorizontalScale();
    const Vector2 patch_size = config.GetPatchSize();
    HeightfieldNoise noise;
    noise.rows = static_cast<int>(patch_size.x() / horizontal_scale);
    noise.cols = static_cast<int>(patch_size.y() / horizontal_scale);
    noise.values.assign(static_cast<std::size_t>(noise.rows * noise.cols), 0);
    const int border = static_cast<int>(terrain.border_width / horizontal_scale);
    const int inner_rows = border > 0 ? noise.rows - 2 * border : noise.rows;
    const int inner_cols = border > 0 ? noise.cols - 2 * border : noise.cols;
    const RealType sample_scale = terrain.downsampled_scale > 0.0
                                          ? terrain.downsampled_scale
                                          : horizontal_scale;
    const int sampled_rows = std::max(2, static_cast<int>(inner_rows * horizontal_scale / sample_scale));
    const int sampled_cols = std::max(2, static_cast<int>(inner_cols * horizontal_scale / sample_scale));
    int minimum = static_cast<int>(terrain.noise_range.x() / kHeightfieldVerticalScale);
    int maximum = static_cast<int>(terrain.noise_range.y() / kHeightfieldVerticalScale);
    if (minimum > maximum) {
        std::swap(minimum, maximum);
    }
    const int step = std::max(1, std::abs(static_cast<int>(terrain.noise_step /
                                                           kHeightfieldVerticalScale)));
    const int choice_count = std::max(1, (maximum - minimum) / step + 1);
    std::vector<int> sampled(static_cast<std::size_t>(sampled_rows * sampled_cols));
    for (int& value : sampled) {
        value = minimum + static_cast<int>(rng->Index(static_cast<std::size_t>(choice_count))) * step;
    }
    const auto sampled_at = [&](int row, int col) {
        return sampled[static_cast<std::size_t>(row * sampled_cols + col)];
    };
    for (int row = 0; row < inner_rows; ++row) {
        const RealType source_row = inner_rows <= 1
                                            ? 0.0
                                            : static_cast<RealType>(row) * (sampled_rows - 1) / (inner_rows - 1);
        const int row0 = static_cast<int>(std::floor(source_row));
        const int row1 = std::min(row0 + 1, sampled_rows - 1);
        const RealType row_alpha = source_row - row0;
        for (int col = 0; col < inner_cols; ++col) {
            const RealType source_col = inner_cols <= 1
                                                ? 0.0
                                                : static_cast<RealType>(col) * (sampled_cols - 1) / (inner_cols - 1);
            const int col0 = static_cast<int>(std::floor(source_col));
            const int col1 = std::min(col0 + 1, sampled_cols - 1);
            const RealType col_alpha = source_col - col0;
            const RealType top = sampled_at(row0, col0) * (1.0 - col_alpha) +
                                 sampled_at(row0, col1) * col_alpha;
            const RealType bottom = sampled_at(row1, col0) * (1.0 - col_alpha) +
                                    sampled_at(row1, col1) * col_alpha;
            NoiseAt(noise, row + border, col + border) =
                    static_cast<int>(std::nearbyint(top * (1.0 - row_alpha) + bottom * row_alpha));
        }
    }
    noise.spawn_height = (terrain.noise_range.x() + terrain.noise_range.y()) * 0.5;
    return noise;
}

HeightfieldNoise MakeWaveNoise(const TerrainGeneratorConfig& config,
                               const TerrainSubTerrainConfig& terrain,
                               RealType difficulty) {
    const RealType horizontal_scale = config.GetHorizontalScale();
    const Vector2 patch_size = config.GetPatchSize();
    HeightfieldNoise noise;
    noise.rows = static_cast<int>(patch_size.x() / horizontal_scale);
    noise.cols = static_cast<int>(patch_size.y() / horizontal_scale);
    noise.values.assign(static_cast<std::size_t>(noise.rows * noise.cols), 0);
    const int border = static_cast<int>(terrain.border_width / horizontal_scale);
    const int inner_rows = border > 0 ? noise.rows - 2 * border : noise.rows;
    const int inner_cols = border > 0 ? noise.cols - 2 * border : noise.cols;
    const RealType amplitude = DifficultyValue(terrain.amplitude_range, difficulty);
    const int amplitude_pixels = static_cast<int>(0.5 * amplitude / kHeightfieldVerticalScale);
    const RealType wave_number = 2.0 * std::numbers::pi_v<RealType> /
                                 (static_cast<RealType>(inner_cols) /
                                  std::max<RealType>(terrain.num_waves, 1.0e-6));
    for (int row = 0; row < inner_rows; ++row) {
        for (int col = 0; col < inner_cols; ++col) {
            const RealType value = amplitude_pixels *
                                   (std::cos(col * wave_number) + std::sin(row * wave_number));
            NoiseAt(noise, row + border, col + border) = static_cast<int>(std::nearbyint(value));
        }
    }
    const auto [minimum, maximum] = std::minmax_element(noise.values.begin(), noise.values.end());
    const RealType max_height = std::max(*maximum - *minimum, 1) * kHeightfieldVerticalScale;
    noise.z_offset = -max_height * 0.5;
    noise.base_thickness_ratio = 0.25;
    return noise;
}

Vector3 AddHeightfield(GeneratedTerrainData* result,
                       const TerrainGeneratorConfig& config,
                       const TerrainSubTerrainConfig& terrain,
                       const Vector3& corner,
                       RealType difficulty,
                       StableRng* rng) {
    HeightfieldNoise noise;
    switch (terrain.type) {
        case TerrainSubTerrainType::PyramidSlope:
            noise = MakeSlopeNoise(config, terrain, difficulty);
            break;
        case TerrainSubTerrainType::RandomRough:
            noise = MakeRandomRoughNoise(config, terrain, rng);
            break;
        case TerrainSubTerrainType::Wave:
            noise = MakeWaveNoise(config, terrain, difficulty);
            break;
        case TerrainSubTerrainType::Flat:
        case TerrainSubTerrainType::PyramidStairs:
            break;
    }

    const auto [minimum, maximum] = std::minmax_element(noise.values.begin(), noise.values.end());
    const int elevation_range = std::max(*maximum - *minimum, 1);
    TerrainHeightField heightfield;
    heightfield.center = corner + Vector3{config.GetPatchSize().x() * 0.5f,
                                          config.GetPatchSize().y() * 0.5f,
                                          0.0f};
    heightfield.size = config.GetPatchSize();
    heightfield.rows = noise.rows;
    heightfield.cols = noise.cols;
    heightfield.heights.resize(noise.values.size());
    heightfield.normalized_elevation.resize(noise.values.size());
    for (int row = 0; row < noise.rows; ++row) {
        const int source_row = noise.rows - 1 - row;
        for (int col = 0; col < noise.cols; ++col) {
            const std::size_t destination = static_cast<std::size_t>(row * noise.cols + col);
            const int value = NoiseAt(noise, source_row, col) - *minimum;
            heightfield.heights[destination] = value * kHeightfieldVerticalScale;
            heightfield.normalized_elevation[destination] =
                    static_cast<RealType>(value) / elevation_range;
        }
    }
    heightfield.base_thickness = elevation_range * kHeightfieldVerticalScale *
                                 noise.base_thickness_ratio;
    heightfield.z_offset = noise.z_offset;
    result->heightfields.push_back(std::move(heightfield));
    return {result->heightfields.back().center.x(),
            result->heightfields.back().center.y(),
            noise.spawn_height};
}

const TerrainSubTerrainConfig& ChooseTerrain(const std::vector<TerrainSubTerrainConfig>& terrains,
                                             const std::vector<RealType>& cumulative_weights,
                                             StableRng* rng) {
    const RealType sample = rng->Uniform();
    const auto iter = std::lower_bound(cumulative_weights.begin(), cumulative_weights.end(), sample);
    const std::size_t index = iter == cumulative_weights.end()
                                      ? terrains.size() - 1
                                      : static_cast<std::size_t>(iter - cumulative_weights.begin());
    return terrains[index];
}

bool ValidateConfig(const TerrainGeneratorConfig& config, std::string* error) {
    const auto fail = [error](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (config.GetSchemaVersion() != TerrainGeneratorConfig::kCurrentSchemaVersion) {
        return fail("Unsupported terrain generator schema version " +
                    std::to_string(config.GetSchemaVersion()) + ".");
    }
    if (config.GetNumRows() <= 0 || config.GetNumRows() > 128 ||
        config.GetNumCols() <= 0 || config.GetNumCols() > 128) {
        return fail("Terrain generator rows and columns must be in [1, 128].");
    }
    if ((config.GetPatchSize().array() <= 0.0).any()) {
        return fail("Terrain generator patch_size must be positive.");
    }
    if (config.GetHorizontalScale() <= 0.0) {
        return fail("Terrain generator horizontal_scale must be positive.");
    }
    const int rows = static_cast<int>(config.GetPatchSize().x() / config.GetHorizontalScale());
    const int cols = static_cast<int>(config.GetPatchSize().y() / config.GetHorizontalScale());
    if (rows < 2 || cols < 2 || rows > 2048 || cols > 2048) {
        return fail("Terrain generator heightfield resolution must be in [2, 2048] per axis.");
    }
    return true;
}

} // namespace

bool GenerateTerrain(const TerrainGeneratorConfig& config,
                     GeneratedTerrainData* result,
                     std::string* error) {
    if (result == nullptr) {
        if (error != nullptr) {
            *error = "Terrain generation result is null.";
        }
        return false;
    }
    *result = {};
    if (!ValidateConfig(config, error)) {
        return false;
    }

    std::vector<TerrainSubTerrainConfig> terrains = config.GetSubTerrains();
    if (terrains.empty()) {
        TerrainSubTerrainConfig flat;
        flat.name = "flat";
        terrains.push_back(flat);
    }

    std::vector<RealType> cumulative_weights;
    cumulative_weights.reserve(terrains.size());
    RealType weight_sum = 0.0;
    for (const TerrainSubTerrainConfig& terrain : terrains) {
        weight_sum += std::max<RealType>(terrain.proportion, 0.0);
        cumulative_weights.push_back(weight_sum);
    }
    if (weight_sum <= 0.0) {
        for (std::size_t index = 0; index < cumulative_weights.size(); ++index) {
            cumulative_weights[index] = static_cast<RealType>(index + 1) / cumulative_weights.size();
        }
    } else {
        for (RealType& weight : cumulative_weights) {
            weight /= weight_sum;
        }
    }

    StableRng rng(config.GetSeed());
    const int rows = config.GetNumRows();
    const int cols = config.IsCurriculum()
                             ? static_cast<int>(terrains.size())
                             : config.GetNumCols();
    const Vector2 patch_size = config.GetPatchSize();
    result->spawn_origins.resize(static_cast<std::size_t>(rows * cols));
    const auto generate_patch = [&](int row,
                                    int col,
                                    const TerrainSubTerrainConfig& terrain,
                                    RealType difficulty) {
        const Vector3 corner{
                -rows * patch_size.x() * 0.5f + row * patch_size.x(),
                -cols * patch_size.y() * 0.5f + col * patch_size.y(),
                0.0f,
        };
        Vector3 origin = corner + Vector3{patch_size.x() * 0.5f, patch_size.y() * 0.5f, 0.0f};
        switch (terrain.type) {
            case TerrainSubTerrainType::Flat:
                AddBox(result,
                       {origin.x(), origin.y(), -0.5f},
                       {patch_size.x(), patch_size.y(), 1.0f},
                       kFlat);
                break;
            case TerrainSubTerrainType::PyramidStairs:
                origin = AddStairs(result, config, terrain, corner, difficulty);
                break;
            case TerrainSubTerrainType::PyramidSlope:
            case TerrainSubTerrainType::RandomRough:
            case TerrainSubTerrainType::Wave:
                origin = AddHeightfield(result, config, terrain, corner, difficulty, &rng);
                break;
        }
        result->spawn_origins[static_cast<std::size_t>(row * cols + col)] = origin;
    };

    if (config.IsCurriculum()) {
        // Keep the column-major generation order used by the rough-terrain task.
        // Spawn origins remain row-major for direct row/column indexing.
        for (int col = 0; col < cols; ++col) {
            const TerrainSubTerrainConfig& terrain =
                    terrains[static_cast<std::size_t>(col) % terrains.size()];
            for (int row = 0; row < rows; ++row) {
                const RealType fraction = (row + rng.Uniform()) / rows;
                const RealType difficulty = config.GetDifficultyRange().x() +
                                            (config.GetDifficultyRange().y() -
                                             config.GetDifficultyRange().x()) *
                                                    fraction;
                generate_patch(row, col, terrain, difficulty);
            }
        }
    } else {
        for (int index = 0; index < rows * cols; ++index) {
            const int row = index / cols;
            const int col = index % cols;
            const TerrainSubTerrainConfig& terrain =
                    ChooseTerrain(terrains, cumulative_weights, &rng);
            const RealType difficulty = rng.Uniform(config.GetDifficultyRange().x(),
                                                    config.GetDifficultyRange().y());
            generate_patch(row, col, terrain, difficulty);
        }
    }

    if (config.GetBorderWidth() > 0.0) {
        const Vector2 inner{rows * patch_size.x(), cols * patch_size.y()};
        const Vector2 outer = inner + Vector2::Constant(2.0 * config.GetBorderWidth());
        AddBorderBoxes(result, outer, inner, 1.0f, {0.0f, 0.0f, -0.5f}, Darken(kPlatform, 0.55f));
    }
    return true;
}

} // namespace gobot
