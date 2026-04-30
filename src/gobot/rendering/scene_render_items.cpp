#include "gobot/rendering/scene_render_items.hpp"

#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/mesh.hpp"

namespace gobot {

namespace {

void CollectNodeRenderItems(const Node* node, SceneRenderItems& items) {
    if (node == nullptr) {
        return;
    }

    const auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance && mesh_instance->IsInsideTree() && mesh_instance->IsVisibleInTree()) {
        Ref<Mesh> mesh_resource = mesh_instance->GetMesh();
        if (mesh_resource.IsValid()) {
            VisualMeshRenderItem item;
            item.mesh = mesh_resource->GetRid();
            item.material = mesh_instance->GetMaterial();
            item.model = mesh_instance->GetGlobalTransform().matrix();
            item.surface_color = mesh_instance->GetSurfaceColor();
            items.visual_meshes.push_back(item);
        }
    }

    const auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node);
    if (collision_shape && collision_shape->IsInsideTree() && collision_shape->IsVisibleInTree() &&
        !collision_shape->IsDisabled()) {
        const Ref<Shape3D>& shape = collision_shape->GetShape();
        if (shape.IsValid()) {
            CollisionDebugRenderItem item;
            item.shape = shape;
            item.transform = collision_shape->GetGlobalTransform();
            items.collision_shapes.push_back(item);
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectNodeRenderItems(node->GetChild(static_cast<int>(i)), items);
    }
}

}

SceneRenderItems CollectSceneRenderItems(const Node* scene_root) {
    SceneRenderItems items;
    CollectNodeRenderItems(scene_root, items);
    return items;
}

}
