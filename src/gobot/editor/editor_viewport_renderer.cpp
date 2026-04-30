#include "gobot/editor/editor_viewport_renderer.hpp"

#include "gobot/editor/editor.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"

#include "imgui.h"

#include <array>
#include <cmath>
#include <utility>

namespace gobot {

namespace {

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

void DrawBoxWireframe(Node* node,
                      const Camera3D* camera,
                      const ImVec2& viewport_position,
                      const ImVec2& viewport_size,
                      ImDrawList* draw_list) {
    auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance && mesh_instance->IsInsideTree() && mesh_instance->IsVisibleInTree()) {
        Ref<BoxMesh> box_mesh = dynamic_pointer_cast<BoxMesh>(mesh_instance->GetMesh());
        if (box_mesh.IsValid()) {
            const Vector3 half_size = box_mesh->GetSize() * 0.5f;
            const std::array<Vector3, 8> local_corners = {
                    Vector3{-half_size.x(), -half_size.y(), -half_size.z()},
                    Vector3{ half_size.x(), -half_size.y(), -half_size.z()},
                    Vector3{ half_size.x(),  half_size.y(), -half_size.z()},
                    Vector3{-half_size.x(),  half_size.y(), -half_size.z()},
                    Vector3{-half_size.x(), -half_size.y(),  half_size.z()},
                    Vector3{ half_size.x(), -half_size.y(),  half_size.z()},
                    Vector3{ half_size.x(),  half_size.y(),  half_size.z()},
                    Vector3{-half_size.x(),  half_size.y(),  half_size.z()},
            };
            const std::array<std::pair<int, int>, 12> edges = {
                    std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}, std::pair{3, 0},
                    std::pair{4, 5}, std::pair{5, 6}, std::pair{6, 7}, std::pair{7, 4},
                    std::pair{0, 4}, std::pair{1, 5}, std::pair{2, 6}, std::pair{3, 7},
            };

            const Affine3 transform = mesh_instance->GetGlobalTransform();
            std::array<ImVec2, 8> screen_corners;
            std::array<bool, 8> visible{};
            for (std::size_t i = 0; i < local_corners.size(); ++i) {
                visible[i] = ProjectPoint(camera, transform * local_corners[i],
                                          viewport_position, viewport_size, screen_corners[i]);
            }

            const bool selected = Editor::GetInstance()->GetSelected() == mesh_instance;
            const ImU32 color = selected ? IM_COL32(255, 196, 64, 255) : IM_COL32(96, 184, 255, 220);
            for (const auto& [from, to] : edges) {
                if (visible[from] && visible[to]) {
                    draw_list->AddLine(screen_corners[from], screen_corners[to], color, selected ? 2.0f : 1.5f);
                }
            }
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        DrawBoxWireframe(node->GetChild(static_cast<int>(i)), camera, viewport_position, viewport_size, draw_list);
    }
}

}

void EditorViewportRenderer::Render(const RID& viewport, const Node* scene_root, const Camera3D* camera) {
    if (scene_root) {
        RS::GetInstance()->RenderSceneToViewport(viewport, scene_root, camera);
    }
    RS::GetInstance()->RenderEditorDebugToViewport(viewport, camera, scene_root);
}

void EditorViewportRenderer::RenderOverlay(const Node* scene_root,
                                           const Camera3D* camera,
                                           const ImVec2& viewport_position,
                                           const ImVec2& viewport_size,
                                           ImDrawList* draw_list) {
    if (!scene_root || !camera || !draw_list) {
        return;
    }

    DrawBoxWireframe(const_cast<Node*>(scene_root), camera, viewport_position, viewport_size, draw_list);
}

}
