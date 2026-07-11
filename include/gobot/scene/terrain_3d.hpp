/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "gobot/core/color.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/terrain_generator_config.hpp"

namespace gobot {

struct GOBOT_EXPORT TerrainBox {
    Vector3 center{Vector3::Zero()};
    Vector3 size{Vector3::Ones()};
    Vector3 rotation_degrees{Vector3::Zero()};
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
};

struct GOBOT_EXPORT TerrainHeightField {
    Vector3 center{Vector3::Zero()};
    Vector2 size{1.0, 1.0};
    int rows{0};
    int cols{0};
    // Row-major heights in image/MuJoCo file convention: columns map to +X,
    // row 0 maps to the +Y edge of the heightfield.
    std::vector<RealType> heights;
    std::vector<RealType> normalized_elevation;
    RealType base_thickness{0.1};
    RealType z_offset{0.0};
};

struct GOBOT_EXPORT TerrainMeshPatch {
    Vector3 center{Vector3::Zero()};
    Vector3 rotation_degrees{Vector3::Zero()};
    std::vector<Vector3> vertices;
    std::vector<std::uint32_t> indices;
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
};

enum class TerrainColorMode {
    SurfaceColor,
    HeightRamp,
    Palette,
};

class GOBOT_EXPORT Terrain3D : public Node3D {
    GOBCLASS(Terrain3D, Node3D)

public:
    Terrain3D() = default;

    void ClearTerrain();

    void AddBox(const TerrainBox& box);

    void AddBox(const Vector3& center,
                const Vector3& size,
                const Vector3& rotation_degrees = Vector3::Zero());

    void SetBoxes(const std::vector<TerrainBox>& boxes);

    const std::vector<TerrainBox>& GetBoxes() const;

    const std::vector<TerrainBox>& GetAuthoredBoxes() const;

    void AddHeightField(const TerrainHeightField& heightfield);

    void SetHeightFields(const std::vector<TerrainHeightField>& heightfields);

    const std::vector<TerrainHeightField>& GetHeightFields() const;

    const std::vector<TerrainHeightField>& GetAuthoredHeightFields() const;

    void AddMeshPatch(const TerrainMeshPatch& mesh_patch);

    void SetMeshPatches(const std::vector<TerrainMeshPatch>& mesh_patches);

    const std::vector<TerrainMeshPatch>& GetMeshPatches() const;

    const std::vector<TerrainMeshPatch>& GetAuthoredMeshPatches() const;

    void SetSpawnOrigins(const std::vector<Vector3>& spawn_origins);

    const std::vector<Vector3>& GetSpawnOrigins() const;

    const std::vector<Vector3>& GetAuthoredSpawnOrigins() const;

    void SetGeneratorConfig(const Ref<TerrainGeneratorConfig>& generator_config);

    Ref<TerrainGeneratorConfig> GetGeneratorConfig() const;

    void RegenerateTerrain();

    const std::string& GetGenerationError() const;

    void SetSurfaceColor(const Color& color);

    Color GetSurfaceColor() const;

    void SetColorMode(TerrainColorMode color_mode);

    TerrainColorMode GetColorMode() const;

    void SetHeightLowColor(const Color& color);

    Color GetHeightLowColor() const;

    void SetHeightHighColor(const Color& color);

    Color GetHeightHighColor() const;

    void SetHeightRangeMin(RealType value);

    RealType GetHeightRangeMin() const;

    void SetHeightRangeMax(RealType value);

    RealType GetHeightRangeMax() const;

    void SetFriction(const Vector3& friction);

    const Vector3& GetFriction() const;

    void SetContactType(int contype);

    int GetContactType() const;

    void SetContactAffinity(int conaffinity);

    int GetContactAffinity() const;

    void SetContactDimension(int condim);

    int GetContactDimension() const;

    void SetSolref(const Vector2& solref);

    const Vector2& GetSolref() const;

    void SetSolimp(const std::vector<RealType>& solimp);

    const std::vector<RealType>& GetSolimp() const;

    void SetMargin(RealType margin);

    RealType GetMargin() const;

    void SetGap(RealType gap);

    RealType GetGap() const;

    Ref<ArrayMesh> GetRenderMesh() const;

private:
    void EnsureGenerated() const;

    void InvalidateGeneratedTerrain();

    void MarkMeshDirty();

    void RebuildRenderMesh() const;

    std::vector<TerrainBox> authored_boxes_;
    std::vector<TerrainHeightField> authored_heightfields_;
    std::vector<TerrainMeshPatch> authored_mesh_patches_;
    std::vector<Vector3> authored_spawn_origins_;
    Ref<TerrainGeneratorConfig> generator_config_;
    mutable std::vector<TerrainBox> generated_boxes_;
    mutable std::vector<TerrainHeightField> generated_heightfields_;
    mutable std::vector<TerrainMeshPatch> generated_mesh_patches_;
    mutable std::vector<Vector3> generated_spawn_origins_;
    mutable std::size_t generated_config_hash_{0};
    mutable bool generated_config_hash_valid_{false};
    mutable std::string generation_error_;
    Color surface_color_{0.48f, 0.56f, 0.50f, 1.0f};
    TerrainColorMode color_mode_{TerrainColorMode::HeightRamp};
    Color height_low_color_{0.10f, 0.34f, 0.30f, 1.0f};
    Color height_high_color_{0.62f, 0.44f, 0.25f, 1.0f};
    RealType height_range_min_{0.0};
    RealType height_range_max_{0.0};
    Vector3 friction_{1.0, 0.005, 0.0001};
    int contype_{1};
    int conaffinity_{1};
    int condim_{3};
    Vector2 solref_{0.02, 1.0};
    std::vector<RealType> solimp_{0.9, 0.95, 0.001, 0.5, 2.0};
    RealType margin_{0.0};
    RealType gap_{0.0};

    mutable Ref<ArrayMesh> render_mesh_{nullptr};
    mutable bool render_mesh_dirty_{true};
};

} // namespace gobot
