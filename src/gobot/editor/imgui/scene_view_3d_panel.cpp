/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/


#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/editor/node3d_editor.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/os/os.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/rendering/texture_storage.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui_extension/fonts/MaterialDesign.inl"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_extension/gizmos/ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <array>
#include <cmath>

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

SceneView3DPanel::SceneView3DPanel()
{
    SetName("SceneView3D");
    SetImGuiWindow(ICON_MDI_EYE " Viewer", "scene_view3d");
    SetImGuiWindowFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    SetImGuiStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});

    current_scene_ = nullptr;

    view_port_ = RS::GetInstance()->ViewportCreate();
}

SceneView3DPanel::~SceneView3DPanel() {
    RS::GetInstance()->Free(view_port_);
}

void SceneView3DPanel::OnImGuiContent()
{
    auto* camera_3d = Node3DEditor::GetInstance()->GetCamera3D();

    ToolBar();

    ImGui::SetCursorPos({0.0, 48.0});
    ImVec2 offset = ImGui::GetCursorPos();

    if(!camera_3d) {
        ImGui::End();
        return;
    }

    const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
    const ImVec2 content_max = ImGui::GetWindowContentRegionMax();
    ImVec2 scene_view_size = {content_max.x - content_min.x, content_max.y - offset.y};
    auto scene_view_position = ImGui::GetWindowPos() + offset;

    if (scene_view_size.x <= 0.0f || scene_view_size.y <= 0.0f) {
        return;
    }

    scene_view_size.x -= static_cast<int>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
    scene_view_size.y -= static_cast<int>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

    RealType aspect = static_cast<RealType>(scene_view_size.x) / static_cast<RealType>(scene_view_size.y);

    if(!AlmostEqual(aspect, camera_3d->GetAspect())) {
        camera_3d->SetAspect(aspect);
    }

    scene_view_size.x -= static_cast<int>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
    scene_view_size.y -= static_cast<int>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

    Resize(static_cast<uint32_t>(scene_view_size.x), static_cast<uint32_t>(scene_view_size.y));

    auto* scene_root = Editor::GetInstance()->GetEditedSceneRoot();
    if (scene_root) {
        RS::GetInstance()->RenderSceneToViewport(view_port_, scene_root, camera_3d);
    }

    ImGuiUtilities::Image(RS::GetInstance()->GetRenderTargetColorTextureNativeHandle(view_port_),
                          {scene_view_size.x, scene_view_size.y});

    if (scene_root) {
        DrawBoxWireframe(scene_root, camera_3d, scene_view_position, scene_view_size, ImGui::GetWindowDrawList());
    }

    ImVec2 min_bound = scene_view_position;
    ImVec2 max_bound = { min_bound.x + scene_view_size.x, min_bound.y + scene_view_size.y };

    bool mouse_inside_rect = ImGui::IsMouseHoveringRect(min_bound, max_bound);

    auto* node3d_editor = Node3DEditor::GetInstance();
    node3d_editor->SetNeedUpdateCamera(mouse_inside_rect);

    ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, scene_view_size.x, scene_view_size.y);
    node3d_editor->OnImGuizmo();

}

void SceneView3DPanel::Resize(uint32_t width, uint32_t height) {
    bool resize = false;
    ERR_FAIL_COND_MSG(width == 0 || height == 0, "Scene3D View Dimensions 0");

    if(width_ != width || height_ != height) {
        resize   = true;
        width_  = width;
        height_ = height;
    }

    if(resize) {
        RS::GetInstance()->ViewportSetSize(view_port_, width_, height_);
    }
}

void SceneView3DPanel::ToolBar()
{
    auto node3d_editor = Node3DEditor::GetInstance();
    ImGui::Indent();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    bool selected = false;

    {
        selected = node3d_editor->GetImGuizmoOperation() == 4;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());
        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_CURSOR_DEFAULT))
            node3d_editor->SetImGuizmoOperation(Node3DEditor::InvalidGuizmoOperation());

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Select");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::TRANSLATE;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());
        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_ARROW_ALL))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::TRANSLATE);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Translate");
    }

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::ROTATE;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_ROTATE_3D))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::ROTATE);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Rotate");
    }

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::SCALE;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_ARROW_EXPAND_ALL))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::SCALE);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Scale");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::UNIVERSAL;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_CROP_ROTATE))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::UNIVERSAL);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Universal");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    {
        selected = node3d_editor->GetImGuizmoOperation() == ImGuizmo::BOUNDS;
        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        ImGui::SameLine();
        if(ImGui::Button(ICON_MDI_BORDER_NONE))
            node3d_editor->SetImGuizmoOperation(ImGuizmo::BOUNDS);

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Bounds");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    ImGui::SameLine();
    {
        selected = node3d_editor->SnapGuizmo();

        if(selected)
            ImGui::PushStyleColor(ImGuiCol_Text, ImGuiUtilities::GetSelectedColor());

        if(ImGui::Button(ICON_MDI_MAGNET))
            node3d_editor->SnapGuizmo() = !selected;

        if(selected)
            ImGui::PopStyleColor();
        ImGuiUtilities::Tooltip("Snap");
    }

    ImGui::SameLine();

    ImGui::PopStyleColor();
    ImGui::Unindent();
}

}
