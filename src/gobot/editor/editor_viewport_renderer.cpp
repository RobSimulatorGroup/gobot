#include "gobot/editor/editor_viewport_renderer.hpp"

#include "gobot/editor/editor.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
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

struct HitResult {
    float distance{std::numeric_limits<float>::max()};
    Vector3 point{Vector3::Zero()};
    bool hit{false};
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

Vector3 GetJointInteractionPosition(const Joint3D* joint) {
    if (joint == nullptr) {
        return Vector3::Zero();
    }

    if (joint->GetJointType() == JointType::Prismatic && joint->GetChildCount() > 0) {
        if (auto* child_3d = Object::PointerCastTo<Node3D>(joint->GetChild(0))) {
            return child_3d->GetGlobalPosition();
        }
    }

    return joint->GetGlobalPosition();
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
                          HitResult& hit_result) {
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
    hit_result.distance = static_cast<float>((world_hit - ray.origin).norm());
    hit_result.point = world_hit;
    hit_result.hit = true;
    return true;
}

bool IntersectLocalSphere(const Ray& ray,
                          const Affine3& transform,
                          RealType radius,
                          HitResult& hit_result) {
    if (!ray.valid || radius <= 0.0) {
        return false;
    }

    const Affine3 inverse_transform = transform.inverse();
    const Vector3 local_origin = inverse_transform * ray.origin;
    const Vector3 local_direction = (inverse_transform.linear() * ray.direction).normalized();
    const RealType b = local_origin.dot(local_direction);
    const RealType c = local_origin.squaredNorm() - radius * radius;
    const RealType discriminant = b * b - c;
    if (discriminant < 0.0) {
        return false;
    }

    RealType t = -b - std::sqrt(discriminant);
    if (t < 0.0) {
        t = -b + std::sqrt(discriminant);
    }
    if (t < 0.0) {
        return false;
    }

    const Vector3 world_hit = transform * (local_origin + local_direction * t);
    hit_result.distance = static_cast<float>((world_hit - ray.origin).norm());
    hit_result.point = world_hit;
    hit_result.hit = true;
    return true;
}

bool IntersectLocalCylinder(const Ray& ray,
                            const Affine3& transform,
                            RealType radius,
                            RealType height,
                            HitResult& hit_result) {
    if (!ray.valid || radius <= 0.0 || height <= 0.0) {
        return false;
    }

    const Affine3 inverse_transform = transform.inverse();
    const Vector3 local_origin = inverse_transform * ray.origin;
    const Vector3 local_direction = (inverse_transform.linear() * ray.direction).normalized();
    const RealType half_height = height * 0.5;
    RealType best_t = std::numeric_limits<RealType>::max();

    const RealType a = local_direction.x() * local_direction.x() + local_direction.z() * local_direction.z();
    const RealType b = 2.0 * (local_origin.x() * local_direction.x() + local_origin.z() * local_direction.z());
    const RealType c = local_origin.x() * local_origin.x() + local_origin.z() * local_origin.z() - radius * radius;
    const RealType discriminant = b * b - 4.0 * a * c;
    if (a > CMP_EPSILON && discriminant >= 0.0) {
        const RealType sqrt_discriminant = std::sqrt(discriminant);
        for (const RealType t : {(-b - sqrt_discriminant) / (2.0 * a),
                                 (-b + sqrt_discriminant) / (2.0 * a)}) {
            const RealType y = local_origin.y() + local_direction.y() * t;
            if (t >= 0.0 && y >= -half_height && y <= half_height) {
                best_t = std::min(best_t, t);
            }
        }
    }

    if (std::abs(local_direction.y()) > CMP_EPSILON) {
        for (const RealType cap_y : {-half_height, half_height}) {
            const RealType t = (cap_y - local_origin.y()) / local_direction.y();
            const Vector3 p = local_origin + local_direction * t;
            if (t >= 0.0 && p.x() * p.x() + p.z() * p.z() <= radius * radius) {
                best_t = std::min(best_t, t);
            }
        }
    }

    if (best_t == std::numeric_limits<RealType>::max()) {
        return false;
    }

    const Vector3 world_hit = transform * (local_origin + local_direction * best_t);
    hit_result.distance = static_cast<float>((world_hit - ray.origin).norm());
    hit_result.point = world_hit;
    hit_result.hit = true;
    return true;
}

bool IntersectLocalTriangleMesh(const Ray& ray,
                                const Affine3& transform,
                                const ArrayMesh* mesh,
                                HitResult& hit_result) {
    if (!ray.valid || mesh == nullptr) {
        return false;
    }

    const std::vector<Vector3>& vertices = mesh->GetVertices();
    const std::vector<uint32_t>& indices = mesh->GetIndices();
    if (vertices.empty()) {
        return false;
    }

    const Affine3 inverse_transform = transform.inverse();
    const Vector3 local_origin = inverse_transform * ray.origin;
    const Vector3 local_direction = (inverse_transform.linear() * ray.direction).normalized();

    bool hit = false;
    float best_distance = std::numeric_limits<float>::max();
    Vector3 best_point = Vector3::Zero();
    auto test_triangle = [&](uint32_t ia, uint32_t ib, uint32_t ic) {
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size()) {
            return;
        }

        const Vector3& v0 = vertices[ia];
        const Vector3& v1 = vertices[ib];
        const Vector3& v2 = vertices[ic];
        const Vector3 edge1 = v1 - v0;
        const Vector3 edge2 = v2 - v0;
        const Vector3 pvec = local_direction.cross(edge2);
        const RealType det = edge1.dot(pvec);
        if (std::abs(det) <= CMP_EPSILON) {
            return;
        }

        const RealType inv_det = 1.0 / det;
        const Vector3 tvec = local_origin - v0;
        const RealType u = tvec.dot(pvec) * inv_det;
        if (u < 0.0 || u > 1.0) {
            return;
        }

        const Vector3 qvec = tvec.cross(edge1);
        const RealType v = local_direction.dot(qvec) * inv_det;
        if (v < 0.0 || u + v > 1.0) {
            return;
        }

        const RealType t = edge2.dot(qvec) * inv_det;
        if (t < 0.0) {
            return;
        }

        const Vector3 world_hit = transform * (local_origin + local_direction * t);
        const float distance = static_cast<float>((world_hit - ray.origin).norm());
        if (distance < best_distance) {
            best_distance = distance;
            best_point = world_hit;
            hit = true;
        }
    };

    if (!indices.empty()) {
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            test_triangle(indices[i], indices[i + 1], indices[i + 2]);
        }
    } else {
        for (std::size_t i = 0; i + 2 < vertices.size(); i += 3) {
            test_triangle(static_cast<uint32_t>(i),
                          static_cast<uint32_t>(i + 1),
                          static_cast<uint32_t>(i + 2));
        }
    }

    if (!hit) {
        return false;
    }

    hit_result.distance = best_distance;
    hit_result.point = best_point;
    hit_result.hit = true;
    return true;
}

HitResult IntersectMesh(const Ray& ray, MeshInstance3D* mesh_instance, bool surface_only) {
    HitResult hit_result;
    if (mesh_instance == nullptr) {
        return hit_result;
    }

    const Ref<Mesh>& mesh = mesh_instance->GetMesh();
    if (!mesh.IsValid()) {
        return hit_result;
    }

    if (Ref<ArrayMesh> array_mesh = dynamic_pointer_cast<ArrayMesh>(mesh); array_mesh.IsValid()) {
        if (IntersectLocalTriangleMesh(ray, mesh_instance->GetGlobalTransform(), array_mesh.Get(), hit_result)) {
            return hit_result;
        }
        if (surface_only) {
            return hit_result;
        }
    }

    if (Ref<SphereMesh> sphere_mesh = dynamic_pointer_cast<SphereMesh>(mesh); sphere_mesh.IsValid()) {
        IntersectLocalSphere(ray, mesh_instance->GetGlobalTransform(), sphere_mesh->GetRadius(), hit_result);
        return hit_result;
    }

    if (Ref<CylinderMesh> cylinder_mesh = dynamic_pointer_cast<CylinderMesh>(mesh); cylinder_mesh.IsValid()) {
        IntersectLocalCylinder(ray,
                               mesh_instance->GetGlobalTransform(),
                               cylinder_mesh->GetRadius(),
                               cylinder_mesh->GetHeight(),
                               hit_result);
        return hit_result;
    }

    if (surface_only && !dynamic_pointer_cast<BoxMesh>(mesh).IsValid()) {
        return hit_result;
    }

    IntersectLocalBounds(ray, mesh_instance->GetGlobalTransform(), GetMeshLocalBounds(mesh), hit_result);
    return hit_result;
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
                       const Node* selected_node,
                       bool show_joint_handles) {
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

    auto* joint = show_joint_handles ? Object::PointerCastTo<Joint3D>(node) : nullptr;
    if (joint && joint->IsInsideTree() && joint->IsVisibleInTree()) {
        ImVec2 joint_screen;
        const Vector3 interaction_position = GetJointInteractionPosition(joint);
        if (ProjectPoint(camera, interaction_position, viewport_position, viewport_size, joint_screen)) {
            const bool selected = selected_node == joint;
            const bool hovered = hovered_node == joint;
            const bool motion_target = motion_target_node == joint;
            const ImU32 color = selected ? IM_COL32(255, 196, 64, 255)
                                         : (hovered || motion_target) ? IM_COL32(96, 184, 255, 255)
                                                                     : IM_COL32(220, 220, 220, 190);
            draw_list->AddCircleFilled(joint_screen, kJointHandleRadiusPixels, color, 16);
            draw_list->AddCircle(joint_screen, kJointPickRadiusPixels, color, 24,
                                 selected || hovered || motion_target ? 2.0f : 1.0f);

            const Vector3 axis_end = interaction_position
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
                          draw_list, hovered_node, motion_target_node, selected_node, show_joint_handles);
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
                       Vector3& best_joint_hit,
                       Node*& best_mesh,
                       float& best_mesh_distance,
                       Vector3& best_mesh_hit,
                       bool surface_only) {
    auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance && mesh_instance->IsInsideTree() && mesh_instance->IsVisibleInTree()) {
        const HitResult hit = IntersectMesh(ray, mesh_instance, surface_only);
        if (hit.hit && hit.distance < best_mesh_distance) {
            best_mesh = mesh_instance;
            best_mesh_distance = hit.distance;
            best_mesh_hit = hit.point;
        }
    }

    auto* joint = surface_only ? nullptr : Object::PointerCastTo<Joint3D>(node);
    if (joint && joint->IsInsideTree() && joint->IsVisibleInTree()) {
        ImVec2 screen;
        const Vector3 interaction_position = GetJointInteractionPosition(joint);
        if (ProjectPoint(camera, interaction_position, viewport_position, viewport_size, screen)) {
            const float dx = screen.x - mouse_position.x;
            const float dy = screen.y - mouse_position.y;
            if (dx * dx + dy * dy <= kJointPickRadiusPixels * kJointPickRadiusPixels) {
                const float distance = static_cast<float>((interaction_position - ray.origin).norm());
                if (distance < best_joint_distance) {
                    best_joint = joint;
                    best_joint_distance = distance;
                    best_joint_hit = interaction_position;
                }
            }
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        PickNodeRecursive(node->GetChild(static_cast<int>(i)), camera, ray,
                          viewport_position, viewport_size, mouse_position,
                          best_joint, best_joint_distance, best_joint_hit,
                          best_mesh, best_mesh_distance, best_mesh_hit, surface_only);
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
                                       const ImVec2& mouse_position,
                                       Vector3* hit_point,
                                       bool prefer_mesh,
                                       bool surface_only) const {
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
    Vector3 best_joint_hit = Vector3::Zero();
    Vector3 best_mesh_hit = Vector3::Zero();
    PickNodeRecursive(scene_root, camera, ray, viewport_position, viewport_size, mouse_position,
                      best_joint, best_joint_distance, best_joint_hit,
                      best_mesh, best_mesh_distance, best_mesh_hit, surface_only);
    if (prefer_mesh && best_mesh != nullptr) {
        if (hit_point != nullptr) {
            *hit_point = best_mesh_hit;
        }
        return best_mesh;
    }
    if (surface_only) {
        if (hit_point != nullptr) {
            *hit_point = best_mesh_hit;
        }
        return best_mesh;
    }
    if (best_joint) {
        if (hit_point != nullptr) {
            *hit_point = best_joint_hit;
        }
        return best_joint;
    }
    if (hit_point != nullptr) {
        *hit_point = best_mesh_hit;
    }
    return best_mesh;
}

void EditorViewportRenderer::RenderOverlay(const Node* scene_root,
                                           const Camera3D* camera,
                                           const ImVec2& viewport_position,
                                           const ImVec2& viewport_size,
                                           ImDrawList* draw_list,
                                           const Node* hovered_node,
                                           const Node* motion_target_node,
                                           bool show_joint_handles) {
    if (!scene_root || !camera || !draw_list) {
        return;
    }

    DrawEditorOverlay(const_cast<Node*>(scene_root), camera, viewport_position, viewport_size, draw_list,
                      hovered_node, show_joint_handles ? motion_target_node : nullptr,
                      Editor::GetInstance()->GetSelected(), show_joint_handles);
}

}
