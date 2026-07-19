#include "gobot/rendering/scene_render_items.hpp"

#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/environment_3d.hpp"
#include "gobot/scene/light_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/terrain_3d.hpp"

#include "gobot/log.hpp"

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
#include <unordered_map>
#endif
#include <algorithm>
#include <bit>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace gobot {

namespace {

Affine3 ResolveNodeRenderTransform(const Node3D* node, const Affine3& parent_transform) {
    if (node == nullptr) {
        return parent_transform;
    }
    return node->IsInsideTree() ? node->GetGlobalTransform() : parent_transform * node->GetTransform();
}

bool IsNodeVisibleForRender(const Node3D* node) {
    return node != nullptr && (node->IsInsideTree() ? node->IsVisibleInTree() : node->IsVisible());
}

std::uint64_t HashCombine(std::uint64_t seed, std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed ^ value;
}

template <typename T>
std::uint64_t HashScalar(std::uint64_t seed, const T& value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        seed ^= bytes[i];
        seed *= 1099511628211ULL;
    }
    return seed;
}

std::uint64_t HashColor(std::uint64_t seed, const Color& color) {
    seed = HashScalar(seed, color.red());
    seed = HashScalar(seed, color.green());
    seed = HashScalar(seed, color.blue());
    return HashScalar(seed, color.alpha());
}

template <typename MatrixType>
std::uint64_t HashMatrix(std::uint64_t seed, const MatrixType& matrix) {
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
        seed = HashScalar(seed, matrix.data()[i]);
    }
    return seed;
}

RenderImageSnapshot CaptureImage(const Ref<Image>& image) {
    RenderImageSnapshot snapshot;
    if (!image.IsValid()) {
        return snapshot;
    }

    struct CachedImageStorage {
        std::uint64_t revision = 0;
        std::weak_ptr<const ImageStorageData> storage;
    };
    static std::mutex cache_mutex;
    static std::unordered_map<ObjectID, CachedImageStorage> cache;

    snapshot.image_id = image->GetInstanceId();
    snapshot.revision = image->GetRevision();
    std::lock_guard lock(cache_mutex);
    CachedImageStorage& cached = cache[snapshot.image_id];
    snapshot.storage = cached.storage.lock();
    if (cached.revision != snapshot.revision || snapshot.storage == nullptr) {
        snapshot.storage = std::make_shared<const ImageStorageData>(image->GetStorageData());
        cached.revision = snapshot.revision;
        cached.storage = snapshot.storage;
    }
    return snapshot;
}

RenderTextureSnapshot CaptureTexture(const Ref<Texture2D>& texture) {
    RenderTextureSnapshot snapshot;
    if (!texture.IsValid()) {
        return snapshot;
    }
    snapshot.texture_id = texture->GetInstanceId();
    snapshot.revision = texture->GetRevision();
    snapshot.image = CaptureImage(texture->GetImage());
    snapshot.min_filter = texture->GetMinFilter();
    snapshot.mag_filter = texture->GetMagFilter();
    snapshot.wrap_u = texture->GetWrapU();
    snapshot.wrap_v = texture->GetWrapV();
    return snapshot;
}

RenderMaterialSnapshot CaptureMaterial(const Ref<Material>& material, const Color& fallback_color) {
    RenderMaterialSnapshot snapshot;
    snapshot.albedo = fallback_color;
    if (!material.IsValid()) {
        return snapshot;
    }
    snapshot.material_id = material->GetInstanceId();
    snapshot.revision = material->GetRevision();
    const Ref<PBRMaterial3D> pbr = dynamic_pointer_cast<PBRMaterial3D>(material);
    if (!pbr.IsValid()) {
        return snapshot;
    }
    snapshot.albedo = pbr->GetAlbedo();
    snapshot.metallic = pbr->GetMetallic();
    snapshot.roughness = pbr->GetRoughness();
    snapshot.specular = pbr->GetSpecular();
    snapshot.albedo_texture = CaptureTexture(pbr->GetAlbedoTexture());
    snapshot.metallic_roughness_texture = CaptureTexture(pbr->GetMetallicRoughnessTexture());
    snapshot.normal_texture = CaptureTexture(pbr->GetNormalTexture());
    snapshot.normal_scale = pbr->GetNormalScale();
    snapshot.occlusion_texture = CaptureTexture(pbr->GetOcclusionTexture());
    snapshot.occlusion_strength = pbr->GetOcclusionStrength();
    snapshot.emissive = pbr->GetEmissive();
    snapshot.emissive_texture = CaptureTexture(pbr->GetEmissiveTexture());
    snapshot.alpha_mode = pbr->GetAlphaMode();
    snapshot.alpha_cutoff = pbr->GetAlphaCutoff();
    snapshot.double_sided = pbr->IsDoubleSided();
    return snapshot;
}

std::uint64_t HashTexture(std::uint64_t seed, const RenderTextureSnapshot& texture) {
    seed = HashCombine(seed, texture.texture_id.operator std::uint64_t());
    seed = HashCombine(seed, texture.revision);
    seed = HashCombine(seed, texture.image.image_id.operator std::uint64_t());
    seed = HashCombine(seed, texture.image.revision);
    seed = HashCombine(seed, static_cast<std::uint64_t>(texture.min_filter));
    seed = HashCombine(seed, static_cast<std::uint64_t>(texture.mag_filter));
    seed = HashCombine(seed, static_cast<std::uint64_t>(texture.wrap_u));
    return HashCombine(seed, static_cast<std::uint64_t>(texture.wrap_v));
}

std::uint64_t HashMaterial(const RenderMaterialSnapshot& material) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = HashCombine(hash, material.material_id.operator std::uint64_t());
    hash = HashCombine(hash, material.revision);
    hash = HashColor(hash, material.albedo);
    hash = HashScalar(hash, material.metallic);
    hash = HashScalar(hash, material.roughness);
    hash = HashScalar(hash, material.specular);
    hash = HashTexture(hash, material.albedo_texture);
    hash = HashTexture(hash, material.metallic_roughness_texture);
    hash = HashTexture(hash, material.normal_texture);
    hash = HashScalar(hash, material.normal_scale);
    hash = HashTexture(hash, material.occlusion_texture);
    hash = HashScalar(hash, material.occlusion_strength);
    hash = HashColor(hash, material.emissive);
    hash = HashTexture(hash, material.emissive_texture);
    hash = HashCombine(hash, static_cast<std::uint64_t>(material.alpha_mode));
    hash = HashScalar(hash, material.alpha_cutoff);
    return HashCombine(hash, material.double_sided ? 1ULL : 0ULL);
}

void AppendMeshItems(SceneRenderItems& items,
                     ObjectID instance_id,
                     const Ref<Mesh>& mesh,
                     const Matrix4& model,
                     const Ref<Material>& instance_material,
                     const Ref<Material>& mesh_material,
                     const Color& fallback_color) {
    const std::shared_ptr<const MeshSurfaceList> surfaces = mesh->GetSurfaceData();
    if (surfaces == nullptr) {
        return;
    }
    for (std::size_t surface_index = 0; surface_index < surfaces->size(); ++surface_index) {
        const MeshSurfaceData& surface = (*surfaces)[surface_index];
        if (surface.vertices.empty() || surface.indices.empty()) {
            continue;
        }
        const Ref<Material> active_material = instance_material.IsValid()
                                                      ? instance_material
                                                      : (surface.material.IsValid() ? surface.material : mesh_material);
        VisualMeshRenderItem item;
        item.instance_id = instance_id;
        item.mesh_id = mesh->GetInstanceId();
        item.mesh_revision = mesh->GetRevision();
        item.surface_index = surface_index;
        item.surfaces = surfaces;
        item.model = model;
        item.material = CaptureMaterial(active_material, fallback_color);
        items.visual_meshes.emplace_back(std::move(item));
    }
}

void CollectLighting(const Node* node,
                     SceneRenderSnapshot& snapshot,
                     const Affine3& parent_transform,
                     bool* found_environment) {
    if (node == nullptr) {
        return;
    }
    const auto* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 node_transform = ResolveNodeRenderTransform(node_3d, parent_transform);

    if (!*found_environment) {
        const auto* environment = Object::PointerCastTo<Environment3D>(node);
        if (environment != nullptr && environment->IsEnabled() && IsNodeVisibleForRender(environment)) {
            snapshot.environment.clear_color = environment->GetClearColor();
            snapshot.environment.sky_color = environment->GetSkyColor();
            snapshot.environment.ground_color = environment->GetGroundColor();
            snapshot.environment.ambient_intensity = environment->GetAmbientIntensity();
            snapshot.environment.exposure = environment->GetExposure();
            snapshot.environment.environment_texture = CaptureTexture(environment->GetEnvironmentTexture());
            snapshot.environment.environment_rotation = node_transform.linear();
            snapshot.environment.environment_intensity = environment->GetEnvironmentIntensity();
            *found_environment = true;
        }
    }

    const auto* light = Object::PointerCastTo<Light3D>(node);
    if (light != nullptr && light->IsEnabled() && IsNodeVisibleForRender(light)) {
        RenderLightSnapshot render_light;
        render_light.light_id = light->GetInstanceId();
        render_light.color = light->GetColor();
        render_light.intensity = light->GetIntensity();
        render_light.position = node_transform.translation();
        if (Object::PointerCastTo<DirectionalLight3D>(light) != nullptr) {
            render_light.type = RenderLightType::Directional;
            render_light.direction = (node_transform.linear() * Vector3::UnitZ()).normalized();
        } else if (const auto* spot = Object::PointerCastTo<SpotLight3D>(light); spot != nullptr) {
            render_light.type = RenderLightType::Spot;
            render_light.direction = (node_transform.linear() * -Vector3::UnitZ()).normalized();
            render_light.range = spot->GetRange();
            render_light.inner_angle = spot->GetInnerAngle();
            render_light.outer_angle = spot->GetOuterAngle();
        } else if (const auto* point = Object::PointerCastTo<PointLight3D>(light); point != nullptr) {
            render_light.type = RenderLightType::Point;
            render_light.range = point->GetRange();
        }
        snapshot.lights.emplace_back(render_light);
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectLighting(node->GetChild(static_cast<int>(i)), snapshot, node_transform, found_environment);
    }
}

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
std::string BuildMaterialDebugSignature(const std::string& material_source,
                                        const Ref<Material>& material,
                                        const Color& color,
                                        RealType metallic,
                                        RealType roughness,
                                        RealType specular) {
    return fmt::format("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
                       material_source,
                       material.IsValid() ? material->GetClassStringName() : std::string("<none>"),
                       material.IsValid() ? material->GetPath() : std::string(),
                       color.red(),
                       color.green(),
                       color.blue(),
                       color.alpha(),
                       metallic,
                       roughness,
                       specular);
}
#endif

void CollectNodeRenderItems(const Node* node, SceneRenderItems& items, const Affine3& parent_transform) {
    if (node == nullptr) {
        return;
    }

    const auto* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 node_transform = ResolveNodeRenderTransform(node_3d, parent_transform);

    const auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance && IsNodeVisibleForRender(mesh_instance)) {
        Ref<Mesh> mesh_resource = mesh_instance->GetMesh();
        if (mesh_resource.IsValid()) {
            const Ref<Material> material = mesh_instance->GetMaterial().IsValid()
                                                   ? mesh_instance->GetMaterial()
                                                   : mesh_instance->GetMeshMaterial();
#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
            std::string material_source = "surface_or_surface_color";
            if (mesh_instance->GetMaterial().IsValid()) {
                material_source = "material";
            } else if (mesh_instance->GetMeshMaterial().IsValid()) {
                material_source = "mesh.material";
            }
#endif
            AppendMeshItems(items,
                            mesh_instance->GetInstanceId(),
                            mesh_resource,
                            node_transform.matrix(),
                            mesh_instance->GetMaterial(),
                            mesh_instance->GetMeshMaterial(),
                            mesh_instance->GetSurfaceColor());
#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
            if (!items.visual_meshes.empty()) {
                const RenderMaterialSnapshot& render_material = items.visual_meshes.back().material;
                static std::unordered_map<const MeshInstance3D*, std::string> logged_material_signatures;
                const std::string signature = BuildMaterialDebugSignature(material_source,
                                                                          material,
                                                                          render_material.albedo,
                                                                          render_material.metallic,
                                                                          render_material.roughness,
                                                                          render_material.specular);
                auto [it, inserted] = logged_material_signatures.emplace(mesh_instance, signature);
                if (inserted || it->second != signature) {
                    it->second = signature;
                    LOG_TRACE("Viewer material node '{}' source '{}' material '{}' path '{}' albedo=({}, {}, {}, {}) metallic={} roughness={} specular={}.",
                              mesh_instance->GetName(),
                              material_source,
                              material.IsValid() ? material->GetClassStringName() : std::string("<none>"),
                              material.IsValid() ? material->GetPath() : std::string(),
                              render_material.albedo.red(),
                              render_material.albedo.green(),
                              render_material.albedo.blue(),
                              render_material.albedo.alpha(),
                              render_material.metallic,
                              render_material.roughness,
                              render_material.specular);
                }
            }
#endif
        }
    }

    const auto* terrain = Object::PointerCastTo<Terrain3D>(node);
    if (terrain && IsNodeVisibleForRender(terrain)) {
        Ref<ArrayMesh> mesh_resource = terrain->GetRenderMesh();
        if (mesh_resource.IsValid()) {
            const bool uses_vertex_colors = terrain->GetColorMode() != TerrainColorMode::SurfaceColor;
            const Color surface_color = uses_vertex_colors
                                                ? Color{1.0f, 1.0f, 1.0f, terrain->GetSurfaceColor().alpha()}
                                                : terrain->GetSurfaceColor();
            const std::size_t first_item = items.visual_meshes.size();
            AppendMeshItems(items,
                            terrain->GetInstanceId(),
                            mesh_resource,
                            node_transform.matrix(),
                            {},
                            mesh_resource->GetMaterial(),
                            surface_color);
            for (std::size_t i = first_item; i < items.visual_meshes.size(); ++i) {
                items.visual_meshes[i].material.roughness = 0.92;
                items.visual_meshes[i].material.specular = 0.12;
            }
        }
    }

    const auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node);
    if (collision_shape &&
        IsNodeVisibleForRender(collision_shape) &&
        !collision_shape->IsDisabled()) {
        const Ref<Shape3D>& shape = collision_shape->GetShape();
        if (shape.IsValid()) {
            CollisionDebugRenderItem item;
            item.shape = shape;
            item.transform = node_transform;
            items.collision_shapes.push_back(item);
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectNodeRenderItems(node->GetChild(static_cast<int>(i)), items, node_transform);
    }
}

}

SceneRenderItems CollectSceneRenderItems(const Node* scene_root) {
    SceneRenderItems items;
    CollectNodeRenderItems(scene_root, items, Affine3::Identity());
    return items;
}

SceneRenderSnapshot CaptureSceneRenderSnapshot(const Node* scene_root,
                                               const Camera3D& camera) {
    SceneRenderItems items = CollectSceneRenderItems(scene_root);

    SceneRenderSnapshot snapshot;
    snapshot.visual_meshes = std::move(items.visual_meshes);
    snapshot.camera.view = camera.GetViewMatrix();
    snapshot.camera.projection = camera.GetProjectionMatrix();
    snapshot.camera.view_projection = snapshot.camera.projection * snapshot.camera.view;
    snapshot.camera.world_position = camera.GetViewMatrixEye();
    bool found_environment = false;
    CollectLighting(scene_root, snapshot, Affine3::Identity(), &found_environment);
    if (snapshot.lights.empty()) {
        RenderLightSnapshot fallback;
        fallback.direction = snapshot.environment.directional_light_direction.normalized();
        fallback.color = snapshot.environment.directional_light_color;
        fallback.intensity = snapshot.environment.directional_light_intensity;
        snapshot.lights.emplace_back(fallback);
    } else {
        const auto directional = std::find_if(
                snapshot.lights.begin(), snapshot.lights.end(),
                [](const RenderLightSnapshot& light) { return light.type == RenderLightType::Directional; });
        if (directional != snapshot.lights.end()) {
            snapshot.environment.directional_light_direction = directional->direction;
            snapshot.environment.directional_light_color = directional->color;
            snapshot.environment.directional_light_intensity = directional->intensity;
        }
    }
    constexpr std::uint64_t seed = 1469598103934665603ULL;
    snapshot.fingerprints.topology = seed;
    snapshot.fingerprints.geometry = seed;
    snapshot.fingerprints.transforms = seed;
    snapshot.fingerprints.materials = seed;
    for (const VisualMeshRenderItem& item : snapshot.visual_meshes) {
        snapshot.fingerprints.topology = HashCombine(
                snapshot.fingerprints.topology, item.instance_id.operator std::uint64_t());
        snapshot.fingerprints.topology = HashCombine(
                snapshot.fingerprints.topology, item.mesh_id.operator std::uint64_t());
        snapshot.fingerprints.topology = HashCombine(snapshot.fingerprints.topology, item.surface_index);
        snapshot.fingerprints.geometry = HashCombine(
                snapshot.fingerprints.geometry, item.mesh_id.operator std::uint64_t());
        snapshot.fingerprints.geometry = HashCombine(snapshot.fingerprints.geometry, item.mesh_revision);
        snapshot.fingerprints.transforms = HashMatrix(snapshot.fingerprints.transforms, item.model);
        snapshot.fingerprints.materials = HashCombine(
                snapshot.fingerprints.materials, HashMaterial(item.material));
    }
    snapshot.fingerprints.camera = HashMatrix(seed, snapshot.camera.view_projection);
    snapshot.fingerprints.camera = HashMatrix(snapshot.fingerprints.camera, snapshot.camera.view);
    snapshot.fingerprints.lighting = HashColor(seed, snapshot.environment.clear_color);
    snapshot.fingerprints.lighting = HashColor(snapshot.fingerprints.lighting, snapshot.environment.sky_color);
    snapshot.fingerprints.lighting = HashColor(snapshot.fingerprints.lighting, snapshot.environment.ground_color);
    snapshot.fingerprints.lighting = HashColor(
            snapshot.fingerprints.lighting, snapshot.environment.directional_light_color);
    snapshot.fingerprints.lighting = HashMatrix(
            snapshot.fingerprints.lighting, snapshot.environment.directional_light_direction);
    snapshot.fingerprints.lighting = HashScalar(
            snapshot.fingerprints.lighting, snapshot.environment.directional_light_intensity);
    snapshot.fingerprints.lighting = HashScalar(
            snapshot.fingerprints.lighting, snapshot.environment.ambient_intensity);
    snapshot.fingerprints.lighting = HashScalar(
            snapshot.fingerprints.lighting, snapshot.environment.exposure);
    snapshot.fingerprints.lighting = HashTexture(
            snapshot.fingerprints.lighting, snapshot.environment.environment_texture);
    snapshot.fingerprints.lighting = HashMatrix(
            snapshot.fingerprints.lighting, snapshot.environment.environment_rotation);
    snapshot.fingerprints.lighting = HashScalar(
            snapshot.fingerprints.lighting, snapshot.environment.environment_intensity);
    for (const RenderLightSnapshot& light : snapshot.lights) {
        snapshot.fingerprints.lighting = HashCombine(
                snapshot.fingerprints.lighting, light.light_id.operator std::uint64_t());
        snapshot.fingerprints.lighting = HashCombine(
                snapshot.fingerprints.lighting, static_cast<std::uint64_t>(light.type));
        snapshot.fingerprints.lighting = HashMatrix(snapshot.fingerprints.lighting, light.position);
        snapshot.fingerprints.lighting = HashMatrix(snapshot.fingerprints.lighting, light.direction);
        snapshot.fingerprints.lighting = HashColor(snapshot.fingerprints.lighting, light.color);
        snapshot.fingerprints.lighting = HashScalar(snapshot.fingerprints.lighting, light.intensity);
        snapshot.fingerprints.lighting = HashScalar(snapshot.fingerprints.lighting, light.range);
        snapshot.fingerprints.lighting = HashScalar(snapshot.fingerprints.lighting, light.inner_angle);
        snapshot.fingerprints.lighting = HashScalar(snapshot.fingerprints.lighting, light.outer_angle);
    }
    snapshot.fingerprints.combined = snapshot.fingerprints.topology;
    snapshot.fingerprints.combined = HashCombine(
            snapshot.fingerprints.combined, snapshot.fingerprints.geometry);
    snapshot.fingerprints.combined = HashCombine(
            snapshot.fingerprints.combined, snapshot.fingerprints.transforms);
    snapshot.fingerprints.combined = HashCombine(
            snapshot.fingerprints.combined, snapshot.fingerprints.materials);
    snapshot.fingerprints.combined = HashCombine(
            snapshot.fingerprints.combined, snapshot.fingerprints.camera);
    snapshot.fingerprints.combined = HashCombine(
            snapshot.fingerprints.combined, snapshot.fingerprints.lighting);
    return snapshot;
}

}
