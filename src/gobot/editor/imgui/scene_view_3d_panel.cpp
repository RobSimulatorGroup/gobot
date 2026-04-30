/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/


#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/editor/node3d_editor.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/editor_viewport_renderer.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui_extension/fonts/MaterialDesign.inl"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_extension/gizmos/ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <memory>

namespace gobot {

namespace {

Robot3D* FindRobotAncestor(Node* node) {
    Node* current = node;
    while (current) {
        if (auto* robot = Object::PointerCastTo<Robot3D>(current)) {
            return robot;
        }
        current = current->GetParent();
    }
    return nullptr;
}

bool IsJointMotionEditable(Joint3D* joint) {
    if (!joint) {
        return false;
    }

    auto* robot = FindRobotAncestor(joint);
    return !robot || robot->GetMode() == RobotMode::Motion;
}

Joint3D* FindNearestAncestorJoint(Node* node) {
    Node* current = node;
    while (current) {
        if (auto* joint = Object::PointerCastTo<Joint3D>(current)) {
            return joint;
        }
        current = current->GetParent();
    }
    return nullptr;
}

Joint3D* FindMotionJointForViewportTarget(Node* target) {
    if (!target) {
        return nullptr;
    }

    if (auto* joint = Object::PointerCastTo<Joint3D>(target)) {
        return IsJointMotionEditable(joint) ? joint : nullptr;
    }

    auto* robot = FindRobotAncestor(target);
    if (!robot || robot->GetMode() != RobotMode::Motion) {
        return nullptr;
    }

    auto* parent_joint = FindNearestAncestorJoint(target->GetParent());
    return IsJointMotionEditable(parent_joint) ? parent_joint : nullptr;
}

}

SceneView3DPanel::SceneView3DPanel()
{
    SetName("SceneView3D");
    SetImGuiWindow(ICON_MDI_EYE " Viewer", "scene_view3d");
    SetImGuiWindowFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    SetImGuiStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});

    current_scene_ = nullptr;
    viewport_renderer_ = std::make_unique<EditorViewportRenderer>();

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
        return;
    }

    const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
    const ImVec2 content_max = ImGui::GetWindowContentRegionMax();
    ImVec2 scene_view_size = {content_max.x - content_min.x, content_max.y - offset.y};

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
    viewport_renderer_->Render(view_port_, scene_root, camera_3d);

    const ImVec2 scene_view_position = ImGui::GetCursorScreenPos();
    ImGuiUtilities::Image(RS::GetInstance()->GetRenderTargetColorTextureNativeHandle(view_port_),
                          {scene_view_size.x, scene_view_size.y},
                          {0.0f, 1.0f},
                          {1.0f, 0.0f});

    ImVec2 min_bound = scene_view_position;
    ImVec2 max_bound = { min_bound.x + scene_view_size.x, min_bound.y + scene_view_size.y };

    bool mouse_inside_rect = ImGui::IsMouseHoveringRect(min_bound, max_bound);

    auto* node3d_editor = Node3DEditor::GetInstance();
    ProcessViewportInput(scene_root, scene_view_position, scene_view_size, mouse_inside_rect);
    node3d_editor->SetNeedUpdateCamera(mouse_inside_rect && !dragged_joint_);

    viewport_renderer_->RenderOverlay(scene_root, camera_3d, scene_view_position, scene_view_size,
                                      ImGui::GetWindowDrawList(), hovered_node_,
                                      dragged_joint_ ? dragged_joint_ : motion_target_joint_);

    ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, scene_view_size.x, scene_view_size.y);
    node3d_editor->OnImGuizmo();

}

void SceneView3DPanel::ProcessViewportInput(Node* scene_root,
                                            const ImVec2& viewport_position,
                                            const ImVec2& viewport_size,
                                            bool mouse_inside_rect) {
    auto* node3d_editor = Node3DEditor::GetInstance();
    if (!scene_root || !mouse_inside_rect || ImGuizmo::IsUsing()) {
        hovered_node_ = nullptr;
        motion_target_joint_ = nullptr;
    } else {
        hovered_node_ = viewport_renderer_->PickNode(scene_root, node3d_editor->GetCamera3D(),
                                                     viewport_position, viewport_size, ImGui::GetMousePos());
    }

    const bool left_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    const bool camera_modifier_down = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift;
    auto* motion_joint = FindMotionJointForViewportTarget(hovered_node_);
    motion_target_joint_ = motion_joint;

    if (!left_down) {
        pressed_joint_ = nullptr;
        dragged_joint_ = nullptr;
        node3d_editor->SetBlockCameraInput(false);
        return;
    }

    if (mouse_inside_rect && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        !camera_modifier_down && !ImGuizmo::IsOver()) {
        if (hovered_node_) {
            Editor::GetInstance()->SetSelected(hovered_node_);
        }

        if (motion_joint) {
            pressed_joint_ = motion_joint;
            dragged_joint_ = motion_joint;
            drag_start_joint_position_ = static_cast<float>(dragged_joint_->GetJointPosition());
            drag_start_mouse_ = ImGui::GetIO().MouseClickedPos[ImGuiMouseButton_Left];
        }
    }

    const bool can_capture_joint = mouse_inside_rect && IsJointMotionEditable(pressed_joint_) && !camera_modifier_down &&
                                   !ImGuizmo::IsUsing() && !ImGuizmo::IsOver();

    if (!dragged_joint_ && can_capture_joint && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0f)) {
        dragged_joint_ = pressed_joint_;
        drag_start_joint_position_ = static_cast<float>(dragged_joint_->GetJointPosition());
        drag_start_mouse_ = ImGui::GetIO().MouseClickedPos[ImGuiMouseButton_Left];
    }

    if (dragged_joint_) {
        const ImVec2 mouse_delta{ImGui::GetMousePos().x - drag_start_mouse_.x,
                                 ImGui::GetMousePos().y - drag_start_mouse_.y};
        constexpr float joint_drag_sensitivity = 0.01f;
        const float signed_delta = mouse_delta.x - mouse_delta.y;
        const JointType joint_type = dragged_joint_->GetJointType();
        if (joint_type == JointType::Revolute || joint_type == JointType::Continuous ||
            joint_type == JointType::Prismatic) {
            dragged_joint_->SetJointPosition(drag_start_joint_position_ + signed_delta * joint_drag_sensitivity);
        }
    }

    node3d_editor->SetBlockCameraInput(can_capture_joint || dragged_joint_);
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
