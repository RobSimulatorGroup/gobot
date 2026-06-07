#include "gobot/rendering/scene_render_items.hpp"

#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/terrain_3d.hpp"

#include "gobot/log.hpp"

#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
#include <unordered_map>
#endif

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
            VisualMeshRenderItem item;
            item.mesh = mesh_resource->GetRid();
            if (!item.mesh.IsNull()) {
                item.material = mesh_instance->GetActiveMaterial();
#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
                std::string material_source = "surface_color";
                if (mesh_instance->GetMaterial().IsValid()) {
                    material_source = "material";
                } else if (mesh_instance->GetMeshMaterial().IsValid()) {
                    material_source = "mesh.material";
                }
#endif
                item.model = node_transform.matrix();
                item.surface_color = mesh_instance->GetSurfaceColor();
                if (Ref<PBRMaterial3D> pbr_material = dynamic_pointer_cast<PBRMaterial3D>(item.material);
                    pbr_material.IsValid()) {
                    item.surface_color = pbr_material->GetAlbedo();
                    item.metallic = pbr_material->GetMetallic();
                    item.roughness = pbr_material->GetRoughness();
                    item.specular = pbr_material->GetSpecular();
                }
#if GOB_LOG_ACTIVE_LEVEL <= GOB_LOG_LEVEL_TRACE
                static std::unordered_map<const MeshInstance3D*, std::string> logged_material_signatures;
                const std::string signature = BuildMaterialDebugSignature(material_source,
                                                                          item.material,
                                                                          item.surface_color,
                                                                          item.metallic,
                                                                          item.roughness,
                                                                          item.specular);
                auto [it, inserted] = logged_material_signatures.emplace(mesh_instance, signature);
                if (inserted || it->second != signature) {
                    it->second = signature;
                    LOG_TRACE("Viewer material node '{}' source '{}' material '{}' path '{}' albedo=({}, {}, {}, {}) metallic={} roughness={} specular={}.",
                              mesh_instance->GetName(),
                              material_source,
                              item.material.IsValid() ? item.material->GetClassStringName() : std::string("<none>"),
                              item.material.IsValid() ? item.material->GetPath() : std::string(),
                              item.surface_color.red(),
                              item.surface_color.green(),
                              item.surface_color.blue(),
                              item.surface_color.alpha(),
                              item.metallic,
                              item.roughness,
                              item.specular);
                }
#endif
                items.visual_meshes.push_back(item);
            }
        }
    }

    const auto* terrain = Object::PointerCastTo<Terrain3D>(node);
    if (terrain && IsNodeVisibleForRender(terrain)) {
        Ref<ArrayMesh> mesh_resource = terrain->GetRenderMesh();
        if (mesh_resource.IsValid()) {
            VisualMeshRenderItem item;
            item.mesh = mesh_resource->GetRid();
            if (!item.mesh.IsNull()) {
                item.model = node_transform.matrix();
                item.surface_color = terrain->GetColorMode() == TerrainColorMode::HeightRamp
                        ? Color{1.0f, 1.0f, 1.0f, terrain->GetSurfaceColor().alpha()}
                        : terrain->GetSurfaceColor();
                item.roughness = 0.92f;
                item.specular = 0.12f;
                items.visual_meshes.push_back(item);
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

}
