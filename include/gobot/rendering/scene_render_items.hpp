/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/core/io/image.hpp"
#include "gobot/core/math/geometry.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/object_id.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/shape_3d.hpp"
#include "gobot/scene/resources/texture.hpp"

#include "gobot_export.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gobot {

class Camera3D;
class Node;
class Terrain3D;

struct RenderImageSnapshot {
    ObjectID image_id;
    std::uint64_t revision = 0;
    std::shared_ptr<const ImageStorageData> storage;

    [[nodiscard]] bool IsValid() const {
        return image_id.IsValid() && storage != nullptr && storage->width > 0 && storage->height > 0;
    }
};

struct RenderTextureSnapshot {
    ObjectID texture_id;
    std::uint64_t revision = 0;
    RenderImageSnapshot image;
    TextureFilter min_filter = TextureFilter::Linear;
    TextureFilter mag_filter = TextureFilter::Linear;
    TextureWrap wrap_u = TextureWrap::Repeat;
    TextureWrap wrap_v = TextureWrap::Repeat;

    [[nodiscard]] bool IsValid() const { return texture_id.IsValid() && image.IsValid(); }
};

struct RenderMaterialSnapshot {
    ObjectID material_id;
    std::uint64_t revision = 0;
    Color albedo{0.66f, 0.78f, 0.95f, 1.0f};
    RealType metallic = 0.0;
    RealType roughness = 0.5;
    RealType specular = 0.5;
    RenderTextureSnapshot albedo_texture;
    RenderTextureSnapshot metallic_roughness_texture;
    RenderTextureSnapshot normal_texture;
    RealType normal_scale = 1.0;
    RenderTextureSnapshot occlusion_texture;
    RealType occlusion_strength = 1.0;
    Color emissive{0.0f, 0.0f, 0.0f, 1.0f};
    RenderTextureSnapshot emissive_texture;
    AlphaMode alpha_mode = AlphaMode::Opaque;
    RealType alpha_cutoff = 0.5;
    bool double_sided = false;
};

struct VisualMeshRenderItem {
    ObjectID instance_id;
    ObjectID mesh_id;
    std::uint64_t mesh_revision = 0;
    std::size_t surface_index = 0;
    std::shared_ptr<const MeshSurfaceList> surfaces;
    Matrix4 model = Matrix4::Identity();
    RenderMaterialSnapshot material;

    [[nodiscard]] const MeshSurfaceData* GetSurface() const {
        return surfaces != nullptr && surface_index < surfaces->size()
                       ? &(*surfaces)[surface_index]
                       : nullptr;
    }
};

struct CollisionDebugRenderItem {
    Ref<Shape3D> shape;
    Affine3 transform = Affine3::Identity();
};

struct SceneRenderItems {
    std::vector<VisualMeshRenderItem> visual_meshes;
    std::vector<CollisionDebugRenderItem> collision_shapes;
};

struct RenderCameraSnapshot {
    Matrix4 view = Matrix4::Identity();
    Matrix4 projection = Matrix4::Identity();
    Matrix4 view_projection = Matrix4::Identity();
    Vector3 world_position = Vector3::Zero();
};

struct RenderEnvironmentSnapshot {
    Color clear_color{0.075f, 0.08f, 0.09f, 1.0f};
    Color sky_color{0.58f, 0.67f, 0.78f, 1.0f};
    Color ground_color{0.09f, 0.095f, 0.105f, 1.0f};
    Color directional_light_color{1.0f, 0.96f, 0.9f, 1.0f};
    Vector3 directional_light_direction{0.35, 0.45, 0.82};
    RealType directional_light_intensity = 1.4;
    RealType ambient_intensity = 0.38;
    RealType exposure = 0.9;
    RenderTextureSnapshot environment_texture;
    Matrix3 environment_rotation = Matrix3::Identity();
    RealType environment_intensity = 1.0;
};

enum class RenderLightType {
    Directional,
    Point,
    Spot
};

struct RenderLightSnapshot {
    ObjectID light_id;
    RenderLightType type = RenderLightType::Directional;
    Vector3 position = Vector3::Zero();
    Vector3 direction = Vector3::UnitZ();
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
    RealType intensity = 1.0;
    RealType range = 0.0;
    RealType inner_angle = 0.0;
    RealType outer_angle = 0.0;
};

struct SceneRenderFingerprints {
    std::uint64_t topology = 0;
    std::uint64_t geometry = 0;
    std::uint64_t transforms = 0;
    std::uint64_t materials = 0;
    std::uint64_t camera = 0;
    std::uint64_t lighting = 0;
    std::uint64_t combined = 0;
};

// Immutable per-frame input shared by raster and future compute renderers.
// Backends must not traverse or retain the authored Scene tree.
struct SceneRenderSnapshot {
    std::vector<VisualMeshRenderItem> visual_meshes;
    RenderCameraSnapshot camera;
    RenderEnvironmentSnapshot environment;
    std::vector<RenderLightSnapshot> lights;
    SceneRenderFingerprints fingerprints;
};

GOBOT_EXPORT SceneRenderItems CollectSceneRenderItems(const Node* scene_root);
GOBOT_EXPORT SceneRenderSnapshot CaptureSceneRenderSnapshot(const Node* scene_root,
                                                             const Camera3D& camera);

}
