#include "gobot/rendering/scene_debug_data.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot/scene/deformable_body_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/capsule_shape_3d.hpp"
#include "gobot/scene/resources/convex_mesh_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace gobot {
namespace {
void PushWorldVertex(std::vector<float>& vertices, const Vector3& point) {
    vertices.push_back(static_cast<float>(point.x()));
    vertices.push_back(static_cast<float>(point.y()));
    vertices.push_back(static_cast<float>(point.z()));
}

void AppendLine(std::vector<float>& vertices,
                const Affine3& transform,
                const Vector3& from,
                const Vector3& to) {
    PushWorldVertex(vertices, transform * from);
    PushWorldVertex(vertices, transform * to);
}

void AppendWorldLine(std::vector<float>& vertices, const Vector3& from, const Vector3& to) {
    PushWorldVertex(vertices, from);
    PushWorldVertex(vertices, to);
}

Affine3 ResolveDebugTransform(const Node3D* node, const Affine3& parent_transform) {
    if (node == nullptr) {
        return parent_transform;
    }
    return node->IsInsideTree()
                   ? node->GetGlobalTransform()
                   : parent_transform * node->GetTransform();
}

bool IsDebugNodeVisible(const Node3D* node, bool parent_visible) {
    if (!parent_visible || node == nullptr) {
        return parent_visible;
    }
    return node->IsInsideTree() ? node->IsVisibleInTree() : node->IsVisible();
}

void CollectDeformableGeometry(const Node* node,
                               const Affine3& parent_transform,
                               bool parent_visible,
                               std::vector<DeformableDebugGeometry>& geometries) {
    if (node == nullptr) {
        return;
    }

    const auto* node_3d = Object::PointerCastTo<Node3D>(node);
    const Affine3 transform = ResolveDebugTransform(node_3d, parent_transform);
    const bool visible = IsDebugNodeVisible(node_3d, parent_visible);
    if (visible) {
        if (const auto* body = Object::PointerCastTo<DeformableBody3D>(node)) {
            const std::vector<Vector3>* authored_vertices = nullptr;
            std::vector<std::uint32_t> surface;
            if (body->GetModel() == DeformableBodyModel::ThinShell) {
                const Ref<SurfaceMesh>& mesh = body->GetSurfaceMesh();
                if (mesh.IsValid()) {
                    authored_vertices = &mesh->GetVertices();
                    surface = mesh->GetTriangles();
                }
            } else {
                const Ref<TetrahedralMesh>& mesh = body->GetMesh();
                if (mesh.IsValid()) {
                    authored_vertices = &mesh->GetVertices();
                    surface = mesh->GetResolvedSurfaceTriangles();
                }
            }
            if (authored_vertices != nullptr) {
                DeformableDebugGeometry geometry;
                geometry.surface_color = body->GetDebugSurfaceColor();
                const std::vector<Vector3>& runtime_vertices = body->GetRuntimeVertices();
                const std::vector<Vector3>& vertices =
                        runtime_vertices.size() == authored_vertices->size()
                                ? runtime_vertices
                                : *authored_vertices;
                for (std::size_t index = 0; index + 2 < surface.size(); index += 3) {
                    const std::uint32_t ia = surface[index];
                    const std::uint32_t ib = surface[index + 1];
                    const std::uint32_t ic = surface[index + 2];
                    if (ia >= vertices.size() || ib >= vertices.size() ||
                        ic >= vertices.size()) {
                        continue;
                    }
                    const Vector3 a = transform * vertices[ia];
                    const Vector3 b = transform * vertices[ib];
                    const Vector3 c = transform * vertices[ic];
                    PushWorldVertex(geometry.triangles, a);
                    PushWorldVertex(geometry.triangles, b);
                    PushWorldVertex(geometry.triangles, c);
                    if (body->IsDebugWireframeVisible()) {
                        AppendWorldLine(geometry.lines, a, b);
                        AppendWorldLine(geometry.lines, b, c);
                        AppendWorldLine(geometry.lines, c, a);
                    }
                }
                if (!geometry.triangles.empty()) {
                    geometries.push_back(std::move(geometry));
                }
            }
        }
    }

    for (std::size_t index = 0; index < node->GetChildCount(); ++index) {
        CollectDeformableGeometry(
                node->GetChild(static_cast<int>(index)), transform, visible,
                geometries);
    }
}

void AppendBoxLines(std::vector<float>& vertices, const Affine3& transform, const Vector3& size) {
    const Vector3 half = size * 0.5f;
    const std::array<Vector3, 8> corners = {
            Vector3{-half.x(), -half.y(), -half.z()},
            Vector3{ half.x(), -half.y(), -half.z()},
            Vector3{ half.x(),  half.y(), -half.z()},
            Vector3{-half.x(),  half.y(), -half.z()},
            Vector3{-half.x(), -half.y(),  half.z()},
            Vector3{ half.x(), -half.y(),  half.z()},
            Vector3{ half.x(),  half.y(),  half.z()},
            Vector3{-half.x(),  half.y(),  half.z()},
    };
    constexpr std::array<std::pair<int, int>, 12> edges = {
            std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}, std::pair{3, 0},
            std::pair{4, 5}, std::pair{5, 6}, std::pair{6, 7}, std::pair{7, 4},
            std::pair{0, 4}, std::pair{1, 5}, std::pair{2, 6}, std::pair{3, 7},
    };

    for (const auto& [from, to] : edges) {
        AppendLine(vertices, transform, corners[from], corners[to]);
    }
}

void AppendCircleLines(std::vector<float>& vertices,
                       const Affine3& transform,
                       RealType radius,
                       int segments,
                       int axis) {
    for (int i = 0; i < segments; ++i) {
        const RealType a = static_cast<RealType>(2.0 * Math_PI * i / segments);
        const RealType b = static_cast<RealType>(2.0 * Math_PI * ((i + 1) % segments) / segments);

        Vector3 from = Vector3::Zero();
        Vector3 to = Vector3::Zero();
        if (axis == 0) {
            from = Vector3{0.0, std::cos(a) * radius, std::sin(a) * radius};
            to = Vector3{0.0, std::cos(b) * radius, std::sin(b) * radius};
        } else if (axis == 1) {
            from = Vector3{std::cos(a) * radius, 0.0, std::sin(a) * radius};
            to = Vector3{std::cos(b) * radius, 0.0, std::sin(b) * radius};
        } else {
            from = Vector3{std::cos(a) * radius, std::sin(a) * radius, 0.0};
            to = Vector3{std::cos(b) * radius, std::sin(b) * radius, 0.0};
        }
        AppendLine(vertices, transform, from, to);
    }
}

void AppendSphereLines(std::vector<float>& vertices, const Affine3& transform, RealType radius) {
    constexpr int segments = 48;
    AppendCircleLines(vertices, transform, radius, segments, 0);
    AppendCircleLines(vertices, transform, radius, segments, 1);
    AppendCircleLines(vertices, transform, radius, segments, 2);
}

void AppendCylinderLines(std::vector<float>& vertices, const Affine3& transform, RealType radius, RealType height) {
    constexpr int segments = 48;
    const RealType half_height = height * static_cast<RealType>(0.5);

    for (int i = 0; i < segments; ++i) {
        const RealType a = static_cast<RealType>(2.0 * Math_PI * i / segments);
        const RealType b = static_cast<RealType>(2.0 * Math_PI * ((i + 1) % segments) / segments);
        const Vector3 top_from{std::cos(a) * radius, std::sin(a) * radius, half_height};
        const Vector3 top_to{std::cos(b) * radius, std::sin(b) * radius, half_height};
        const Vector3 bottom_from{std::cos(a) * radius, std::sin(a) * radius, -half_height};
        const Vector3 bottom_to{std::cos(b) * radius, std::sin(b) * radius, -half_height};

        AppendLine(vertices, transform, top_from, top_to);
        AppendLine(vertices, transform, bottom_from, bottom_to);

        if (i % 12 == 0) {
            AppendLine(vertices, transform, bottom_from, top_from);
        }
    }
}

void AppendCapsuleLines(std::vector<float>& vertices, const Affine3& transform, RealType radius, RealType height) {
    constexpr int segments = 48;
    const RealType half_height = height * static_cast<RealType>(0.5);

    AppendCylinderLines(vertices, transform, radius, height);

    Affine3 top_transform = transform;
    top_transform.translation() = transform * Vector3{0.0, 0.0, half_height};
    AppendSphereLines(vertices, top_transform, radius);

    Affine3 bottom_transform = transform;
    bottom_transform.translation() = transform * Vector3{0.0, 0.0, -half_height};
    AppendSphereLines(vertices, bottom_transform, radius);
}

void AppendTriangleMeshLines(std::vector<float>& vertices,
                             const Affine3& transform,
                             const Ref<Mesh>& mesh) {
    if (!mesh.IsValid()) {
        return;
    }
    const std::shared_ptr<const MeshSurfaceList> surfaces = mesh->GetSurfaceData();
    if (!surfaces) {
        return;
    }

    for (const MeshSurfaceData& surface : *surfaces) {
        const auto append_triangle = [&](std::size_t ia, std::size_t ib, std::size_t ic) {
            if (ia >= surface.vertices.size() ||
                ib >= surface.vertices.size() ||
                ic >= surface.vertices.size()) {
                return;
            }
            const Vector3& a = surface.vertices[ia];
            const Vector3& b = surface.vertices[ib];
            const Vector3& c = surface.vertices[ic];
            AppendLine(vertices, transform, a, b);
            AppendLine(vertices, transform, b, c);
            AppendLine(vertices, transform, c, a);
        };

        if (surface.indices.empty()) {
            for (std::size_t i = 0; i + 2 < surface.vertices.size(); i += 3) {
                append_triangle(i, i + 1, i + 2);
            }
            continue;
        }
        for (std::size_t i = 0; i + 2 < surface.indices.size(); i += 3) {
            append_triangle(surface.indices[i],
                            surface.indices[i + 1],
                            surface.indices[i + 2]);
        }
    }
}

void CollectCollisionLines(const SceneRenderItems& render_items, std::vector<float>& vertices) {
    for (const CollisionDebugRenderItem& item : render_items.collision_shapes) {
        if (Ref<BoxShape3D> box = dynamic_pointer_cast<BoxShape3D>(item.shape); box.IsValid()) {
            AppendBoxLines(vertices, item.transform, box->GetSize());
        } else if (Ref<SphereShape3D> sphere = dynamic_pointer_cast<SphereShape3D>(item.shape); sphere.IsValid()) {
            AppendSphereLines(vertices, item.transform, static_cast<RealType>(sphere->GetRadius()));
        } else if (Ref<CylinderShape3D> cylinder = dynamic_pointer_cast<CylinderShape3D>(item.shape); cylinder.IsValid()) {
            AppendCylinderLines(vertices,
                                item.transform,
                                static_cast<RealType>(cylinder->GetRadius()),
                                static_cast<RealType>(cylinder->GetHeight()));
        } else if (Ref<CapsuleShape3D> capsule = dynamic_pointer_cast<CapsuleShape3D>(item.shape); capsule.IsValid()) {
            AppendCapsuleLines(vertices,
                               item.transform,
                               static_cast<RealType>(capsule->GetRadius()),
                               static_cast<RealType>(capsule->GetHeight()));
        } else if (Ref<ConvexMeshShape3D> convex_mesh = dynamic_pointer_cast<ConvexMeshShape3D>(item.shape);
                   convex_mesh.IsValid()) {
            AppendTriangleMeshLines(vertices, item.transform, convex_mesh->GetMesh());
        }
    }
}

} // namespace

SceneDebugData CaptureSceneDebugData(const Node* root,
                                     const PhysicsSceneState* state,
                                     const PhysicsWorldSettings& settings,
                                     bool show_collision_shapes) {
    SceneDebugData data;
    data.settings = settings;
    if (show_collision_shapes) {
        CollectCollisionLines(CollectSceneRenderItems(root), data.collision_lines);
    }
    CollectDeformableGeometry(root, Affine3::Identity(), true, data.deformables);
    if (state != nullptr) {
        auto capture_sensor = [&](const PhysicsSensorState& sensor) {
            if (sensor.enabled && sensor.visualize_debug && sensor.visible) {
                data.sensors.push_back(sensor);
            }
        };
        for (const auto& robot : state->robots) {
            for (const auto& sensor : robot.sensors) {
                capture_sensor(sensor);
            }
        }
        for (const auto& sensor : state->loose_sensors) {
            capture_sensor(sensor);
        }
        data.contacts = state->contacts;
    }
    return data;
}
} // namespace gobot
