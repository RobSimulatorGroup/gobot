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
#include "gobot/error_macros.hpp"
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

namespace gobot {

SceneView3DPanel::SceneView3DPanel()
{
    SetName(ICON_MDI_EYE " Viewer###scene_view3d");
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

    auto scene_view_size = ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin() - offset * 0.5f;
    auto scene_view_position = ImGui::GetWindowPos() + offset;

    scene_view_size.x -= static_cast<int>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
    scene_view_size.y -= static_cast<int>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

    RealType aspect = static_cast<RealType>(scene_view_size.x) / static_cast<RealType>(scene_view_size.y);

    if(!AlmostEqual(aspect, camera_3d->GetAspect())) {
        camera_3d->SetAspect(aspect);
    }

    scene_view_size.x -= static_cast<int>(scene_view_size.x) % 2 != 0 ? 1.0f : 0.0f;
    scene_view_size.y -= static_cast<int>(scene_view_size.y) % 2 != 0 ? 1.0f : 0.0f;

    Resize(static_cast<uint32_t>(scene_view_size.x), static_cast<uint32_t>(scene_view_size.y));

    ImGuiUtilities::Image(RS::GetInstance()->GetRenderTargetColorTextureNativeHandle(view_port_),
                          {scene_view_size.x, scene_view_size.y});

    auto window_size = ImGui::GetWindowSize();
    ImVec2 min_bound = scene_view_position;
    ImVec2 max_bound = { min_bound.x + window_size.x, min_bound.y + window_size.y };

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
