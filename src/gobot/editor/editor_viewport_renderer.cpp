#include "gobot/editor/editor_viewport_renderer.hpp"

#include "gobot/editor/editor.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"

#include "imgui.h"

#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace gobot {

namespace {

constexpr float kJointPickRadiusPixels = 14.0f;
constexpr float kJointHandleRadiusPixels = 6.0f;

struct LocalBounds {
    Vector3 min{Vector3::Zero()};
    Vector3 max{Vector3::Zero()};
    bool valid{false};
};

struct Ray {
    Vector3 origin{Vector3::Zero()};
    Vector3 direction{Vector3::UnitX()};
    bool valid{false};
};

bool ProjectPoint(const Camera3D* camera,
                  const Vector3& world,
                  const ImVec2& viewport_position,
                  const ImVec2& viewport_size,
                  ImVec2& screen) {
    Vector4 world_h(world.x(), world.y(), world.z(), 1.0f);
    Vector4 clip = camera->GetProjectionMatrix() * camera->GetViewMatrix() * world_h;
    if (std::abs(clip.w()) <= CMP_EPSILON) {
        return false;
    }

    Vector3 ndc = clip.head<3>() / clip.w();
    if (ndc.z() < -1.0f || ndc.z() > 1.0f) {
        return false;
    }

    screen.x = viewport_position.x + static_cast<float>((ndc.x() + 1.0f) * 0.5f * viewport_size.x);
    screen.y = viewport_position.y + static_cast<float>((1.0f - ndc.y()) * 0.5f * viewport_size.y);
    return true;
}

Ray MakeRay(const Camera3D* camera,
            const ImVec2& viewport_position,
            const ImVec2& viewport_size,
            const ImVec2& mouse_position) {
    if (!camera || viewport_size.x <= 0.0f || viewport_size.y <= 0.0f) {
        return {};
    }

    const float ndc_x = ((mouse_position.x - viewport_position.x) / viewport_size.x) * 2.0f - 1.0f;
    const float ndc_y = 1.0f - ((mouse_position.y - viewport_position.y) / viewport_size.y) * 2.0f;
    const Matrix4 inverse_view_projection = (camera->GetProjectionMatrix() * camera->GetViewMatrix()).inverse();

    Vector4 near_clip(ndc_x, ndc_y, -1.0f, 1.0f);
    Vector4 far_clip(ndc_x, ndc_y, 1.0f, 1.0f);
    Vector4 near_world_h = inverse_view_projection * near_clip;
    Vector4 far_world_h = inverse_view_projection * far_clip;
    if (std::abs(near_world_h.w()) <= CMP_EPSILON || std::abs(far_world_h.w()) <= CMP_EPSILON) {
        return {};
    }

    const Vector3 near_world = near_world_h.head<3>() / near_world_h.w();
    const Vector3 far_world = far_world_h.head<3>() / far_world_h.w();
    Vector3 direction = far_world - near_world;
    if (direction.squaredNorm() <= CMP_EPSILON2) {
        return {};
    }

    direction.normalize();
    return {near_world, direction, true};
}

LocalBounds GetMeshLocalBounds(const Ref<Mesh>& mesh) {
    if (!mesh.IsValid()) {
        return {};
    }

    if (Ref<BoxMesh> box_mesh = dynamic_pointer_cast<BoxMesh>(mesh); box_mesh.IsValid()) {
        const Vector3 half_size = box_mesh->GetSize() * 0.5f;
        return {-half_size, half_size, true};
    }

    if (Ref<SphereMesh> sphere_mesh = dynamic_pointer_cast<SphereMesh>(mesh); sphere_mesh.IsValid()) {
        const RealType radius = sphere_mesh->GetRadius();
        const Vector3 radius_vector{radius, radius, radius};
        return {-radius_vector, radius_vector, true};
    }

    if (Ref<CylinderMesh> cylinder_mesh = dynamic_pointer_cast<CylinderMesh>(mesh); cylinder_mesh.IsValid()) {
        const RealType radius = cylinder_mesh->GetRadius();
        const RealType half_height = cylinder_mesh->GetHeight() * 0.5;
        return {Vector3{-radius, -half_height, -radius}, Vector3{radius, half_height, radius}, true};
    }

    if (Ref<ArrayMesh> array_mesh = dynamic_pointer_cast<ArrayMesh>(mesh); array_mesh.IsValid()) {
        const std::vector<Vector3>& vertices = array_mesh->GetVertices();
        if (vertices.empty()) {
            return {};
        }

        Vector3 min = vertices.front();
        Vector3 max = vertices.front();
        for (const Vector3& vertex : vertices) {
            min = min.cwiseMin(vertex);
            max = max.cwiseMax(vertex);
        }
        return {min, max, true};
    }

    return {};
}

std::array<Vector3, 8> GetBoundsCorners(const LocalBounds& bounds) {
    return {
            Vector3{bounds.min.x(), bounds.min.y(), bounds.min.z()},
            Vector3{bounds.max.x(), bounds.min.y(), bounds.min.z()},
            Vector3{bounds.max.x(), bounds.max.y(), bounds.min.z()},
            Vector3{bounds.min.x(), bounds.max.y(), bounds.min.z()},
            Vector3{bounds.min.x(), bounds.min.y(), bounds.max.z()},
            Vector3{bounds.max.x(), bounds.min.y(), bounds.max.z()},
            Vector3{bounds.max.x(), bounds.max.y(), bounds.max.z()},
            Vector3{bounds.min.x(), bounds.max.y(), bounds.max.z()},
    };
}

bool IntersectLocalBounds(const Ray& ray,
                          const Affine3& transform,
                          const LocalBounds& bounds,
                          float& world_distance) {
    if (!ray.valid || !bounds.valid) {
        return false;
    }

    const Affine3 inverse_transform = transform.inverse();
    const Vector3 local_origin = inverse_transform * ray.origin;
    const Vector3 local_direction = inverse_transform.linear() * ray.direction;

    RealType t_min = 0.0;
    RealType t_max = std::numeric_limits<RealType>::max();
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(local_direction[axis]) <= CMP_EPSILON) {
            if (local_origin[axis] < bounds.min[axis] || local_origin[axis] > bounds.max[axis]) {
                return false;
            }
            continue;
        }

        RealType t1 = (bounds.min[axis] - local_origin[axis]) / local_direction[axis];
        RealType t2 = (bounds.max[axis] - local_origin[axis]) / local_direction[axis];
        if (t1 > t2) {
            std::swap(t1, t2);
        }
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);
        if (t_min > t_max) {
            return false;
        }
    }

    const Vector3 local_hit = local_origin + local_direction * t_min;
    const Vector3 world_hit = transform * local_hit;
    world_distance = static_cast<float>((world_hit - ray.origin).norm());
    return true;
}

void DrawBoundsWireframe(MeshInstance3D* mesh_instance,
                         const LocalBounds& bounds,
                         const Camera3D* camera,
                         const ImVec2& viewport_position,
                         const ImVec2& viewport_size,
                         ImDrawList* draw_list,
                         ImU32 color,
                         float thickness) {
    if (!bounds.valid) {
        return;
    }

    const std::array<std::pair<int, int>, 12> edges = {
            std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}, std::pair{3, 0},
            std::pair{4, 5}, std::pair{5, 6}, std::pair{6, 7}, std::pair{7, 4},
            std::pair{0, 4}, std::pair{1, 5}, std::pair{2, 6}, std::pair{3, 7},
    };

    const Affine3 transform = mesh_instance->GetGlobalTransform();
    const std::array<Vector3, 8> local_corners = GetBoundsCorners(bounds);
    std::array<ImVec2, 8> screen_corners;
    std::array<bool, 8> visible{};
    for (std::size_t i = 0; i < local_corners.size(); ++i) {
        visible[i] = ProjectPoint(camera, transform * local_corners[i],
                                  viewport_position, viewport_size, screen_corners[i]);
    }

    for (const auto& [from, to] : edges) {
        if (visible[from] && visible[to]) {
            draw_list->AddLine(screen_corners[from], screen_corners[to], color, thickness);
        }
    }
}

void DrawEditorOverlay(Node* node,
                       const Camera3D* camera,
                       const ImVec2& viewport_position,
                       const ImVec2& viewport_size,
                       ImDrawList* draw_list,
                       const Node* hovered_node,
                       const Node* motion_target_node,
                       const Node* selected_node) {
    auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance && mesh_instance->IsInsideTree() && mesh_instance->IsVisibleInTree()) {
        const bool selected = selected_node == mesh_instance;
        const bool hovered = hovered_node == mesh_instance;
        if (selected || hovered) {
            const ImU32 color = selected ? IM_COL32(255, 196, 64, 255) : IM_COL32(96, 184, 255, 230);
            DrawBoundsWireframe(mesh_instance, GetMeshLocalBounds(mesh_instance->GetMesh()), camera,
                                viewport_position, viewport_size, draw_list, color, selected ? 2.5f : 2.0f);
        }
    }

    auto* joint = Object::PointerCastTo<Joint3D>(node);
    if (joint && joint->IsInsideTree() && joint->IsVisibleInTree()) {
        ImVec2 joint_screen;
        if (ProjectPoint(camera, joint->GetGlobalPosition(), viewport_position, viewport_size, joint_screen)) {
            const bool selected = selected_node == joint;
            const bool hovered = hovered_node == joint;
            const bool motion_target = motion_target_node == joint;
            const ImU32 color = selected ? IM_COL32(255, 196, 64, 255)
                                         : (hovered || motion_target) ? IM_COL32(96, 184, 255, 255)
                                                                     : IM_COL32(220, 220, 220, 190);
            draw_list->AddCircleFilled(joint_screen, kJointHandleRadiusPixels, color, 16);
            draw_list->AddCircle(joint_screen, kJointPickRadiusPixels, color, 24,
                                 selected || hovered || motion_target ? 2.0f : 1.0f);

            const Vector3 axis_end = joint->GetGlobalPosition()
                    + joint->GetGlobalTransform().linear() * joint->GetAxis().normalized() * 0.22;
            ImVec2 axis_screen;
            if (ProjectPoint(camera, axis_end, viewport_position, viewport_size, axis_screen)) {
                draw_list->AddLine(joint_screen, axis_screen, color,
                                   selected || hovered || motion_target ? 2.0f : 1.0f);
            }
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        DrawEditorOverlay(node->GetChild(static_cast<int>(i)), camera, viewport_position, viewport_size,
                          draw_list, hovered_node, motion_target_node, selected_node);
    }
}

void PickNodeRecursive(Node* node,
                       const Camera3D* camera,
                       const Ray& ray,
                       const ImVec2& viewport_position,
                       const ImVec2& viewport_size,
                       const ImVec2& mouse_position,
                       Node*& best_joint,
                       float& best_joint_distance,
                       Node*& best_mesh,
                       float& best_mesh_distance) {
    auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance && mesh_instance->IsInsideTree() && mesh_instance->IsVisibleInTree()) {
        float distance = 0.0f;
        if (IntersectLocalBounds(ray, mesh_instance->GetGlobalTransform(),
                                 GetMeshLocalBounds(mesh_instance->GetMesh()), distance) &&
            distance < best_mesh_distance) {
            best_mesh = mesh_instance;
            best_mesh_distance = distance;
        }
    }

    auto* joint = Object::PointerCastTo<Joint3D>(node);
    if (joint && joint->IsInsideTree() && joint->IsVisibleInTree()) {
        ImVec2 screen;
        if (ProjectPoint(camera, joint->GetGlobalPosition(), viewport_position, viewport_size, screen)) {
            const float dx = screen.x - mouse_position.x;
            const float dy = screen.y - mouse_position.y;
            if (dx * dx + dy * dy <= kJointPickRadiusPixels * kJointPickRadiusPixels) {
                const float distance = static_cast<float>((joint->GetGlobalPosition() - ray.origin).norm());
                if (distance < best_joint_distance) {
                    best_joint = joint;
                    best_joint_distance = distance;
                }
            }
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        PickNodeRecursive(node->GetChild(static_cast<int>(i)), camera, ray,
                          viewport_position, viewport_size, mouse_position,
                          best_joint, best_joint_distance, best_mesh, best_mesh_distance);
    }
}

}

void EditorViewportRenderer::Render(const RID& viewport, const Node* scene_root, const Camera3D* camera) {
    if (scene_root) {
        RS::GetInstance()->RenderSceneToViewport(viewport, scene_root, camera);
    }
    RS::GetInstance()->RenderEditorDebugToViewport(viewport, camera, scene_root);
}

Node* EditorViewportRenderer::PickNode(Node* scene_root,
                                       const Camera3D* camera,
                                       const ImVec2& viewport_position,
                                       const ImVec2& viewport_size,
                                       const ImVec2& mouse_position) const {
    if (!scene_root || !camera) {
        return nullptr;
    }

    const Ray ray = MakeRay(camera, viewport_position, viewport_size, mouse_position);
    if (!ray.valid) {
        return nullptr;
    }

    Node* best_joint = nullptr;
    Node* best_mesh = nullptr;
    float best_joint_distance = std::numeric_limits<float>::max();
    float best_mesh_distance = std::numeric_limits<float>::max();
    PickNodeRecursive(scene_root, camera, ray, viewport_position, viewport_size, mouse_position,
                      best_joint, best_joint_distance, best_mesh, best_mesh_distance);
    return best_joint ? best_joint : best_mesh;
}

void EditorViewportRenderer::RenderOverlay(const Node* scene_root,
                                           const Camera3D* camera,
                                           const ImVec2& viewport_position,
                                           const ImVec2& viewport_size,
                                           ImDrawList* draw_list,
                                           const Node* hovered_node,
                                           const Node* motion_target_node) {
    if (!scene_root || !camera || !draw_list) {
        return;
    }

    DrawEditorOverlay(const_cast<Node*>(scene_root), camera, viewport_position, viewport_size, draw_list,
                      hovered_node, motion_target_node, Editor::GetInstance()->GetSelected());
}

}
