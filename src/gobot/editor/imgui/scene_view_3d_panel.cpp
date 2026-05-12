/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-10
 * SPDX-License-Identifier: Apache-2.0
 */


#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/editor/node3d_editor.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/editor_viewport_renderer.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/simulation/simulation_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/scene/scene_command.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "imgui_extension/fonts/MaterialDesign.inl"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_extension/gizmos/ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

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

Robot3D* FindActiveRobot(Node* hovered_node) {
    if (auto* robot = FindRobotAncestor(Editor::GetInstance()->GetSelected())) {
        return robot;
    }
    return FindRobotAncestor(hovered_node);
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

bool ProjectWorldPoint(const Camera3D* camera,
                       const ImVec2& viewport_position,
                       const ImVec2& viewport_size,
                       const Vector3& world,
                       ImVec2& screen) {
    Vector4 world_h(world.x(), world.y(), world.z(), 1.0f);
    Vector4 clip = camera->GetProjectionMatrix() * camera->GetViewMatrix() * world_h;
    if (std::abs(clip.w()) <= CMP_EPSILON) {
        return false;
    }

    const Vector3 ndc = clip.head<3>() / clip.w();
    if (ndc.z() < -1.0f || ndc.z() > 1.0f) {
        return false;
    }

    screen.x = viewport_position.x + static_cast<float>((ndc.x() + 1.0f) * 0.5f * viewport_size.x);
    screen.y = viewport_position.y + static_cast<float>((1.0f - ndc.y()) * 0.5f * viewport_size.y);
    return true;
}

Vector3 GetJointInteractionPosition(const Joint3D* joint);

bool GetJointScreenAxis(Joint3D* joint,
                        const Camera3D* camera,
                        const ImVec2& viewport_position,
                        const ImVec2& viewport_size,
                        ImVec2& screen_axis) {
    if (!joint || !camera) {
        return false;
    }

    const Vector3 joint_position = GetJointInteractionPosition(joint);
    const Vector3 world_axis = joint->GetGlobalTransform().linear() * joint->GetAxis().normalized();
    ImVec2 joint_screen;
    ImVec2 axis_screen;
    if (!ProjectWorldPoint(camera, viewport_position, viewport_size, joint_position, joint_screen) ||
        !ProjectWorldPoint(camera, viewport_position, viewport_size, joint_position + world_axis, axis_screen)) {
        return false;
    }

    screen_axis = {axis_screen.x - joint_screen.x, axis_screen.y - joint_screen.y};
    const float length = std::sqrt(screen_axis.x * screen_axis.x + screen_axis.y * screen_axis.y);
    if (length <= 2.0f) {
        return false;
    }

    screen_axis.x /= length;
    screen_axis.y /= length;
    return true;
}

Vector3 CameraPlaneDeltaToWorld(const Camera3D* camera, const ImVec2& mouse_delta, RealType scale) {
    if (camera == nullptr) {
        return Vector3::Zero();
    }

    const Vector3 view_direction = camera->GetViewMatrixAt() - camera->GetViewMatrixEye();
    const Vector3 view_right = view_direction.squaredNorm() > CMP_EPSILON2
            ? view_direction.cross(camera->GetViewMatrixUp()).normalized()
            : Vector3::UnitX();
    const Vector3 view_up = camera->GetViewMatrixUp().normalized();
    return (view_right * static_cast<RealType>(mouse_delta.x) +
            view_up * static_cast<RealType>(-mouse_delta.y)) * scale;
}

RealType MouseDepthScale(const Camera3D* camera,
                         const Vector3& point,
                         const ImVec2& viewport_size) {
    if (camera == nullptr || viewport_size.y <= 0.0f) {
        return 1.0;
    }

    const Vector3 view_direction = camera->GetViewMatrixAt() - camera->GetViewMatrixEye();
    if (view_direction.squaredNorm() <= CMP_EPSILON2) {
        return 1.0;
    }

    const RealType depth = std::max<RealType>(1.0, (point - camera->GetViewMatrixEye()).dot(view_direction.normalized()));
    const RealType fovy_radians = camera->GetFovy() * M_PI / 180.0;
    return 2.0 * depth * std::tan(fovy_radians * 0.5) / std::max<RealType>(1.0, viewport_size.y);
}

Vector3 LinkLocalPoint(Link3D* link, const Vector3& world_point) {
    if (link == nullptr) {
        return Vector3::Zero();
    }
    return link->GetGlobalTransform().inverse() * world_point;
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

float GetJointScreenRotationSign(Joint3D* joint, const Camera3D* camera) {
    if (!joint || !camera) {
        return 1.0f;
    }

    const Vector3 world_axis = joint->GetGlobalTransform().linear() * joint->GetAxis().normalized();
    const Vector3 joint_to_camera = camera->GetViewMatrixEye() - GetJointInteractionPosition(joint);
    if (joint_to_camera.isZero(CMP_EPSILON)) {
        return 1.0f;
    }

    return world_axis.dot(joint_to_camera.normalized()) >= 0.0 ? -1.0f : 1.0f;
}

float AngleAroundScreenPoint(const ImVec2& center, const ImVec2& point) {
    return std::atan2(point.y - center.y, point.x - center.x);
}

float WrappedAngleDelta(float from, float to) {
    float delta = to - from;
    while (delta > static_cast<float>(M_PI)) {
        delta -= static_cast<float>(M_PI * 2.0);
    }
    while (delta < static_cast<float>(-M_PI)) {
        delta += static_cast<float>(M_PI * 2.0);
    }
    return delta;
}

const char* JointTypeName(JointType joint_type) {
    switch (joint_type) {
        case JointType::Fixed:
            return "Fixed";
        case JointType::Revolute:
            return "Revolute";
        case JointType::Continuous:
            return "Continuous";
        case JointType::Prismatic:
            return "Prismatic";
        default:
            return "Joint";
    }
}

bool IsJointAtLowerLimit(const Joint3D* joint) {
    return joint && joint->HasJointPositionLimits() &&
           std::abs(joint->GetJointPosition() - joint->GetLowerLimit()) <= 0.0001;
}

bool IsJointAtUpperLimit(const Joint3D* joint) {
    return joint && joint->HasJointPositionLimits() &&
           std::abs(joint->GetJointPosition() - joint->GetUpperLimit()) <= 0.0001;
}

double DisplayJointPosition(JointType joint_type, RealType value) {
    if (joint_type == JointType::Revolute || joint_type == JointType::Continuous) {
        return static_cast<double>(value * 180.0 / M_PI);
    }

    return static_cast<double>(value);
}

const char* JointPositionUnit(JointType joint_type) {
    if (joint_type == JointType::Revolute || joint_type == JointType::Continuous) {
        return "deg";
    }

    if (joint_type == JointType::Prismatic) {
        return "m";
    }

    return "";
}

bool ImGuiBlocksViewportInput() {
    ImGuiContext* context = ImGui::GetCurrentContext();
    if (context == nullptr) {
        return false;
    }

    return ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
}

bool IsSceneRuntimePlaying() {
    auto* editor = Editor::GetInstanceOrNull();
    auto* simulation = SimulationServer::HasInstance() ? SimulationServer::GetInstance() : nullptr;
    return editor != nullptr &&
           editor->IsScenePlaySessionRunning() &&
           simulation != nullptr &&
           !simulation->IsPaused();
}

Link3D* FindNearestAncestorLink(Node* node) {
    Node* current = node;
    while (current != nullptr) {
        if (auto* link = Object::PointerCastTo<Link3D>(current)) {
            return link;
        }
        current = current->GetParent();
    }
    return nullptr;
}

bool ApplyRuntimeSpringForce(Link3D* link,
                             const Vector3& local_point,
                             const Vector3& target_point,
                             const Vector3& force_hint) {
    if (link == nullptr || !SimulationServer::HasInstance()) {
        return false;
    }

    Robot3D* robot = FindRobotAncestor(link);
    if (robot == nullptr) {
        return false;
    }

    return SimulationServer::GetInstance()->SetLinkSpringForce(robot->GetName(),
                                                              link->GetName(),
                                                              local_point,
                                                              target_point,
                                                              force_hint);
}

void ClearRuntimeExternalForce() {
    if (SimulationServer::HasInstance()) {
        SimulationServer::GetInstance()->ClearExternalForces();
    }
}

}

void DrawJointMotionHint(Joint3D* joint,
                         const Camera3D* camera,
                         const ImVec2& viewport_position,
                         const ImVec2& viewport_size,
                         ImDrawList* draw_list,
                         bool dragging);

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

bool SceneView3DPanel::Begin() {
    if (auto* editor = Editor::GetInstanceOrNull()) {
        SetImGuiWindow(std::string(ICON_MDI_EYE " ") + editor->GetSceneViewTitle(), "scene_view3d");
    }
    return ImGuiWindow::Begin();
}

void SceneView3DPanel::OnImGuiContent()
{
    auto* camera_3d = Node3DEditor::GetInstance()->GetCamera3D();
    if(!camera_3d) {
        return;
    }

    const float viewport_top_offset = 56.0f;
    const ImVec2 window_size = ImGui::GetWindowSize();
    const ImVec2 main_viewport_size = ImGui::GetMainViewport()->Size;
    ImVec2 scene_view_size = {window_size.x, window_size.y - viewport_top_offset};

    if (scene_view_size.x <= 0.0f || scene_view_size.y <= 0.0f) {
        return;
    }

    scene_view_size.x = std::min(scene_view_size.x, main_viewport_size.x);
    scene_view_size.y = std::min(scene_view_size.y, main_viewport_size.y);

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

    auto* scene_root = Editor::GetInstance()->GetActiveSceneRoot();
    viewport_renderer_->Render(view_port_, scene_root, camera_3d);

    ImGui::SetCursorPos({0.0f, viewport_top_offset});
    const ImVec2 scene_view_position = ImGui::GetCursorScreenPos();
    const auto render_texture = static_cast<ImTextureID>(
            reinterpret_cast<std::uintptr_t>(RS::GetInstance()->GetRenderTargetColorTextureNativeHandle(view_port_)));
    ImGuiUtilities::Image(render_texture,
                          {scene_view_size.x, scene_view_size.y},
                          {0.0f, 1.0f},
                          {1.0f, 0.0f});

    ImVec2 min_bound = scene_view_position;
    ImVec2 max_bound = { min_bound.x + scene_view_size.x, min_bound.y + scene_view_size.y };

    auto* node3d_editor = Node3DEditor::GetInstance();
    ImGuizmo::SetRect(scene_view_position.x, scene_view_position.y, scene_view_size.x, scene_view_size.y);
    if (!Editor::GetInstance()->IsScenePlaySessionRunning()) {
        node3d_editor->OnImGuizmo();
    }

    const bool imgui_blocks_viewport_input = ImGuiBlocksViewportInput();
    bool mouse_inside_rect = ImGui::IsMouseHoveringRect(min_bound, max_bound) && !imgui_blocks_viewport_input;
    ProcessViewportInput(scene_root, scene_view_position, scene_view_size, mouse_inside_rect);
    node3d_editor->SetNeedUpdateCamera(mouse_inside_rect && !dragged_joint_ && !drag_force_active_ &&
                                       !ImGuizmo::IsUsing() && !ImGuizmo::IsOver());

    auto* selected_motion_joint = IsSceneRuntimePlaying()
            ? nullptr
            : FindMotionJointForViewportTarget(Editor::GetInstance()->GetSelected());
    auto* active_motion_joint = IsSceneRuntimePlaying()
            ? nullptr
            : (dragged_joint_ ? dragged_joint_
                              : (motion_target_joint_ ? motion_target_joint_ : selected_motion_joint));
    viewport_renderer_->RenderOverlay(scene_root, camera_3d, scene_view_position, scene_view_size,
                                      ImGui::GetWindowDrawList(), hovered_node_, active_motion_joint,
                                      !IsSceneRuntimePlaying());
    DrawJointMotionHint(active_motion_joint,
                        camera_3d,
                        scene_view_position,
                        scene_view_size,
                        ImGui::GetWindowDrawList(),
                        dragged_joint_ != nullptr);
    if (drag_force_active_) {
        ImVec2 start;
        ImVec2 end;
        const Vector3 force_start = dragged_force_link_ != nullptr
                ? dragged_force_link_->GetGlobalTransform() * drag_force_local_point_
                : drag_force_point_;
        if (ProjectWorldPoint(camera_3d, scene_view_position, scene_view_size, force_start, start) &&
            ProjectWorldPoint(camera_3d, scene_view_position, scene_view_size,
                              drag_force_target_point_, end)) {
            auto* draw_list = ImGui::GetWindowDrawList();
            const ImU32 color = IM_COL32(255, 196, 64, 255);
            draw_list->AddCircleFilled(start, 5.0f, color, 16);
            draw_list->AddLine(start, end, color, 3.0f);
            const ImVec2 direction{end.x - start.x, end.y - start.y};
            const float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
            if (length > 1.0f) {
                const ImVec2 unit{direction.x / length, direction.y / length};
                const ImVec2 normal{-unit.y, unit.x};
                const ImVec2 tip = end;
                draw_list->AddTriangleFilled(tip,
                                             {tip.x - unit.x * 12.0f + normal.x * 5.0f,
                                              tip.y - unit.y * 12.0f + normal.y * 5.0f},
                                             {tip.x - unit.x * 12.0f - normal.x * 5.0f,
                                              tip.y - unit.y * 12.0f - normal.y * 5.0f},
                                             color);
            }
        }
    }

    ToolBar({scene_view_position.x + 8.0f, scene_view_position.y + 8.0f});
    ImGui::SetCursorScreenPos({scene_view_position.x, scene_view_position.y + scene_view_size.y});
    ImGui::Dummy({1.0f, 1.0f});

}

void SceneView3DPanel::ProcessViewportInput(Node* scene_root,
                                            const ImVec2& viewport_position,
                                            const ImVec2& viewport_size,
                                            bool mouse_inside_rect) {
    auto* node3d_editor = Node3DEditor::GetInstance();
    const bool gizmo_captures_mouse = ImGuizmo::IsUsing() || ImGuizmo::IsOver();
    const bool runtime_playing = IsSceneRuntimePlaying();
    if (!scene_root || !mouse_inside_rect || gizmo_captures_mouse) {
        hovered_node_ = nullptr;
        motion_target_joint_ = nullptr;
    } else {
        Vector3 hit_point = Vector3::Zero();
        hovered_node_ = viewport_renderer_->PickNode(scene_root, node3d_editor->GetCamera3D(),
                                                     viewport_position, viewport_size, ImGui::GetMousePos(),
                                                     &hit_point, runtime_playing, runtime_playing);
        if (hovered_node_ != nullptr && !drag_force_point_locked_) {
            drag_force_point_ = hit_point;
        }
    }

    const bool left_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    const bool camera_modifier_down = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift;
    auto* motion_joint = runtime_playing ? nullptr : FindMotionJointForViewportTarget(hovered_node_);
    motion_target_joint_ = motion_joint;

    if (!left_down) {
        pressed_joint_ = nullptr;
        dragged_joint_ = nullptr;
        if (drag_force_active_) {
            ClearRuntimeExternalForce();
        }
        dragged_force_link_ = nullptr;
        drag_force_active_ = false;
        drag_force_point_locked_ = false;
        drag_force_joint_ = nullptr;
        drag_force_point_ = Vector3::Zero();
        drag_force_local_point_ = Vector3::Zero();
        drag_force_target_point_ = Vector3::Zero();
        drag_force_vector_ = Vector3::Zero();
        drag_force_world_axis_ = Vector3::UnitX();
        drag_force_scale_ = 1.0;
        drag_force_axis_valid_ = false;
        drag_joint_screen_center_valid_ = false;
        drag_joint_screen_axis_valid_ = false;
        drag_last_angle_valid_ = false;
        drag_joint_rotation_sign_ = 1.0f;
        node3d_editor->SetBlockCameraInput(false);
        return;
    }

    if (mouse_inside_rect && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        !camera_modifier_down && !gizmo_captures_mouse) {
        if (hovered_node_) {
            Editor::GetInstance()->SetSelected(hovered_node_);
        } else {
            Editor::GetInstance()->SetSelected(nullptr);
        }

        if (motion_joint) {
            pressed_joint_ = motion_joint;
            dragged_joint_ = motion_joint;
            drag_last_mouse_ = ImGui::GetIO().MouseClickedPos[ImGuiMouseButton_Left];
            drag_joint_screen_center_valid_ = ProjectWorldPoint(node3d_editor->GetCamera3D(),
                                                                viewport_position,
                                                                viewport_size,
                                                                GetJointInteractionPosition(dragged_joint_),
                                                                drag_joint_screen_center_);
            drag_joint_screen_axis_valid_ = GetJointScreenAxis(dragged_joint_, node3d_editor->GetCamera3D(),
                                                               viewport_position, viewport_size,
                                                               drag_joint_screen_axis_);
            drag_joint_rotation_sign_ = GetJointScreenRotationSign(dragged_joint_, node3d_editor->GetCamera3D());
            drag_last_angle_valid_ = drag_joint_screen_center_valid_;
            if (drag_last_angle_valid_) {
                drag_last_angle_ = AngleAroundScreenPoint(drag_joint_screen_center_, drag_last_mouse_);
            }
        } else if (runtime_playing) {
            if (auto* link = FindNearestAncestorLink(hovered_node_)) {
                dragged_force_link_ = link;
                drag_force_joint_ = FindNearestAncestorJoint(link);
                drag_force_local_point_ = LinkLocalPoint(link, drag_force_point_);
                drag_force_target_point_ = drag_force_point_;
                drag_force_scale_ = MouseDepthScale(node3d_editor->GetCamera3D(), drag_force_point_, viewport_size);
                drag_force_axis_valid_ = false;
                if (drag_force_joint_ != nullptr &&
                    drag_force_joint_->GetJointType() == JointType::Prismatic &&
                    drag_force_joint_->GetAxis().squaredNorm() > CMP_EPSILON2) {
                    drag_force_world_axis_ = (drag_force_joint_->GetGlobalTransform().linear() *
                                             drag_force_joint_->GetAxis().normalized()).normalized();
                    drag_force_axis_valid_ = true;
                } else {
                    drag_force_joint_ = nullptr;
                    drag_force_world_axis_ = Vector3::UnitX();
                }
                drag_last_mouse_ = ImGui::GetIO().MouseClickedPos[ImGuiMouseButton_Left];
                drag_force_active_ = true;
                drag_force_point_locked_ = true;
                drag_force_vector_ = Vector3::Zero();
                ApplyRuntimeSpringForce(dragged_force_link_,
                                        drag_force_local_point_,
                                        drag_force_target_point_,
                                        drag_force_vector_);
            }
        }
    }

    const bool can_capture_joint = mouse_inside_rect && IsJointMotionEditable(pressed_joint_) && !camera_modifier_down &&
                                   !gizmo_captures_mouse && !runtime_playing;

    if (!dragged_joint_ && can_capture_joint && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0f)) {
        dragged_joint_ = pressed_joint_;
        drag_last_mouse_ = ImGui::GetIO().MouseClickedPos[ImGuiMouseButton_Left];
        drag_joint_screen_center_valid_ = ProjectWorldPoint(node3d_editor->GetCamera3D(),
                                                            viewport_position,
                                                            viewport_size,
                                                            GetJointInteractionPosition(dragged_joint_),
                                                            drag_joint_screen_center_);
        drag_joint_screen_axis_valid_ = GetJointScreenAxis(dragged_joint_, node3d_editor->GetCamera3D(),
                                                           viewport_position, viewport_size,
                                                           drag_joint_screen_axis_);
        drag_joint_rotation_sign_ = GetJointScreenRotationSign(dragged_joint_, node3d_editor->GetCamera3D());
        drag_last_angle_valid_ = drag_joint_screen_center_valid_;
        if (drag_last_angle_valid_) {
            drag_last_angle_ = AngleAroundScreenPoint(drag_joint_screen_center_, drag_last_mouse_);
        }
    }

    if (dragged_joint_) {
        const ImVec2 mouse_position = ImGui::GetMousePos();
        const ImVec2 mouse_delta{mouse_position.x - drag_last_mouse_.x,
                                 mouse_position.y - drag_last_mouse_.y};
        const JointType joint_type = dragged_joint_->GetJointType();
        const RealType old_joint_position = dragged_joint_->GetJointPosition();
        if ((joint_type == JointType::Revolute || joint_type == JointType::Continuous) &&
            drag_joint_screen_center_valid_ && drag_last_angle_valid_) {
            const float current_angle = AngleAroundScreenPoint(drag_joint_screen_center_, mouse_position);
            const float delta_angle = WrappedAngleDelta(drag_last_angle_, current_angle);
            if (std::abs(delta_angle) > CMP_EPSILON) {
                const RealType new_joint_position = old_joint_position + drag_joint_rotation_sign_ * delta_angle;
                if (std::abs(new_joint_position - old_joint_position) > CMP_EPSILON) {
                    if (auto* context = Editor::GetInstance()->GetEngineContext()) {
                        context->ExecuteSceneCommand(std::make_unique<SetNodePropertyCommand>(
                                dragged_joint_->GetInstanceId(),
                                "joint_position",
                                Variant(new_joint_position)));
                    }
                }
            }
            drag_last_angle_ = current_angle;
        } else if (joint_type == JointType::Prismatic) {
            const float signed_pixels = drag_joint_screen_axis_valid_
                    ? mouse_delta.x * drag_joint_screen_axis_.x + mouse_delta.y * drag_joint_screen_axis_.y
                    : mouse_delta.x - mouse_delta.y;
            if (std::abs(signed_pixels) > CMP_EPSILON) {
                const RealType new_joint_position = old_joint_position + signed_pixels * 0.005f;
                if (std::abs(new_joint_position - old_joint_position) > CMP_EPSILON) {
                    if (auto* context = Editor::GetInstance()->GetEngineContext()) {
                        context->ExecuteSceneCommand(std::make_unique<SetNodePropertyCommand>(
                                dragged_joint_->GetInstanceId(),
                                "joint_position",
                                Variant(new_joint_position)));
                    }
                }
            }
        }
        drag_last_mouse_ = mouse_position;
    } else if (drag_force_active_ && dragged_force_link_ != nullptr) {
        const ImVec2 mouse_position = ImGui::GetMousePos();
        const ImVec2 mouse_delta{mouse_position.x - drag_last_mouse_.x,
                                 mouse_position.y - drag_last_mouse_.y};
        const Vector3 world_delta = CameraPlaneDeltaToWorld(node3d_editor->GetCamera3D(),
                                                            mouse_delta,
                                                            drag_force_scale_);
        if (drag_force_axis_valid_) {
            drag_force_target_point_ += drag_force_world_axis_ * world_delta.dot(drag_force_world_axis_);
        } else {
            drag_force_target_point_ += world_delta;
        }
        drag_force_point_ = dragged_force_link_->GetGlobalTransform() * drag_force_local_point_;
        drag_force_vector_ = drag_force_target_point_ - drag_force_point_;
        ApplyRuntimeSpringForce(dragged_force_link_,
                                drag_force_local_point_,
                                drag_force_target_point_,
                                drag_force_axis_valid_ ? drag_force_world_axis_ : Vector3::Zero());
        drag_last_mouse_ = mouse_position;
    }

    node3d_editor->SetBlockCameraInput(can_capture_joint || dragged_joint_ || drag_force_active_);
}

void DrawJointMotionHint(Joint3D* joint,
                         const Camera3D* camera,
                         const ImVec2& viewport_position,
                         const ImVec2& viewport_size,
                         ImDrawList* draw_list,
                         bool dragging) {
    if (!joint || !camera || !draw_list) {
        return;
    }

    ImVec2 joint_screen;
    if (!ProjectWorldPoint(camera, viewport_position, viewport_size,
                           GetJointInteractionPosition(joint), joint_screen)) {
        return;
    }

    const JointType joint_type = joint->GetJointType();
    std::vector<std::string> lines;
    if (joint_type == JointType::Revolute || joint_type == JointType::Continuous) {
        lines.emplace_back(fmt::format("{}  {:.1f} {}", JointTypeName(joint_type),
                                       DisplayJointPosition(joint_type, joint->GetJointPosition()),
                                       JointPositionUnit(joint_type)));
    } else if (joint_type == JointType::Prismatic) {
        lines.emplace_back(fmt::format("{}  {:.3f} {}", JointTypeName(joint_type),
                                       DisplayJointPosition(joint_type, joint->GetJointPosition()),
                                       JointPositionUnit(joint_type)));
    } else {
        lines.emplace_back(JointTypeName(joint_type));
    }

    const bool has_limits = joint->HasJointPositionLimits();
    const bool at_lower_limit = IsJointAtLowerLimit(joint);
    const bool at_upper_limit = IsJointAtUpperLimit(joint);
    if (has_limits) {
        if (joint_type == JointType::Revolute) {
            lines.emplace_back(fmt::format("limit  {:.1f} .. {:.1f} deg",
                                           DisplayJointPosition(joint_type, joint->GetLowerLimit()),
                                           DisplayJointPosition(joint_type, joint->GetUpperLimit())));
        } else {
            lines.emplace_back(fmt::format("limit  {:.3f} .. {:.3f} m",
                                           DisplayJointPosition(joint_type, joint->GetLowerLimit()),
                                           DisplayJointPosition(joint_type, joint->GetUpperLimit())));
        }

        if (at_lower_limit) {
            lines.emplace_back("lower limit");
        } else if (at_upper_limit) {
            lines.emplace_back("upper limit");
        }
    }

    ImVec2 text_size{0.0f, 0.0f};
    const float line_gap = 2.0f;
    for (const auto& line : lines) {
        const ImVec2 line_size = ImGui::CalcTextSize(line.c_str());
        text_size.x = std::max(text_size.x, line_size.x);
        text_size.y += line_size.y;
    }
    if (lines.size() > 1) {
        text_size.y += line_gap * static_cast<float>(lines.size() - 1);
    }

    const ImVec2 padding{8.0f, 5.0f};
    const float limit_bar_height = has_limits ? 4.0f : 0.0f;
    ImVec2 box_min{joint_screen.x + 14.0f, joint_screen.y - text_size.y - 18.0f};
    ImVec2 box_max{box_min.x + text_size.x + padding.x * 2.0f,
                   box_min.y + text_size.y + padding.y * 2.0f + limit_bar_height};
    const ImVec2 viewport_min = viewport_position;
    const ImVec2 viewport_max{viewport_position.x + viewport_size.x, viewport_position.y + viewport_size.y};
    if (box_max.x > viewport_max.x - 4.0f) {
        const float shift = box_max.x - (viewport_max.x - 4.0f);
        box_min.x -= shift;
        box_max.x -= shift;
    }
    if (box_min.y < viewport_min.y + 4.0f) {
        const float shift = (viewport_min.y + 4.0f) - box_min.y;
        box_min.y += shift;
        box_max.y += shift;
    }

    const ImU32 box_fill = dragging ? IM_COL32(36, 96, 142, 235)
                                    : IM_COL32(28, 28, 28, 220);
    const ImU32 border_color = (at_lower_limit || at_upper_limit)
            ? IM_COL32(255, 172, 64, 255)
            : (dragging ? IM_COL32(120, 205, 255, 255) : IM_COL32(96, 184, 255, 220));
    draw_list->AddRectFilled(box_min, box_max,
                             box_fill,
                             4.0f);
    draw_list->AddRect(box_min, box_max, border_color, 4.0f);

    float text_y = box_min.y + padding.y;
    for (const auto& line : lines) {
        draw_list->AddText({box_min.x + padding.x, text_y},
                           IM_COL32(245, 245, 245, 255),
                           line.c_str());
        text_y += ImGui::CalcTextSize(line.c_str()).y + line_gap;
    }

    if (has_limits) {
        const float bar_min_x = box_min.x + padding.x;
        const float bar_max_x = box_max.x - padding.x;
        const float bar_y = box_max.y - padding.y - limit_bar_height;
        draw_list->AddRectFilled({bar_min_x, bar_y}, {bar_max_x, bar_y + limit_bar_height},
                                 IM_COL32(70, 70, 70, 255), 2.0f);
        const float t = static_cast<float>((joint->GetJointPosition() - joint->GetLowerLimit()) /
                                           (joint->GetUpperLimit() - joint->GetLowerLimit()));
        const float marker_x = bar_min_x + std::clamp(t, 0.0f, 1.0f) * (bar_max_x - bar_min_x);
        draw_list->AddRectFilled({bar_min_x, bar_y}, {marker_x, bar_y + limit_bar_height},
                                 border_color, 2.0f);
    }
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
    ToolBar(ImGui::GetCursorScreenPos());
}

void SceneView3DPanel::ToolBar(const ImVec2& screen_position)
{
    auto node3d_editor = Node3DEditor::GetInstance();
    auto* active_robot = FindActiveRobot(hovered_node_);
    const float base_font_size = ImGui::GetFontSize();
    const float button_extent = std::max(30.0f, base_font_size * 1.32f);
    const ImVec2 button_size{button_extent, button_extent};
    const float icon_font_size = std::max(18.0f, base_font_size * 1.08f);
    const float separator_height = button_extent * 0.68f;
    const float rounding = 4.0f;
    const float right_limit = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x - 8.0f;
    const float row_start_x = screen_position.x;

    ImGui::SetCursorScreenPos(screen_position);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {4.0f, 4.0f});

    struct ToolbarButtonStyle {
        ImVec2 size;
        float icon_size;
        float rounding;
    };
    const ToolbarButtonStyle toolbar_style{button_size, icon_font_size, rounding};

    auto wrap_toolbar_item = [&](float item_width) {
        const ImVec2 cursor = ImGui::GetCursorScreenPos();
        if (cursor.x > row_start_x && cursor.x + item_width > right_limit) {
            ImGui::NewLine();
            ImGui::SetCursorScreenPos({row_start_x, cursor.y + button_size.y + 4.0f});
        }
    };

    auto draw_toolbar_button = [&](const char* id, const char* icon, const char* tooltip,
                                   bool selected, bool enabled, const auto& on_pressed) {
        wrap_toolbar_item(toolbar_style.size.x);
        if (!enabled) {
            ImGui::BeginDisabled();
        }

        ImGui::PushID(id);
        const bool pressed = ImGui::InvisibleButton("button", toolbar_style.size);
        const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
        const ImVec2 item_min = ImGui::GetItemRectMin();
        const ImVec2 item_max = ImGui::GetItemRectMax();
        auto* draw_list = ImGui::GetWindowDrawList();

        if (selected || hovered) {
            const ImGuiCol bg_color = selected ? ImGuiCol_ButtonActive : ImGuiCol_ButtonHovered;
            draw_list->AddRectFilled(item_min, item_max, ImGui::GetColorU32(bg_color), toolbar_style.rounding);
        }

        const ImVec4 text_color = !enabled
                                  ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled)
                                  : (selected ? ImVec4(ImGuiUtilities::GetSelectedColor())
                                              : ImGui::GetStyleColorVec4(ImGuiCol_Text));
        ImFont* font = ImGui::GetFont();
        const ImVec2 text_size = font->CalcTextSizeA(toolbar_style.icon_size,
                                                     std::numeric_limits<float>::max(),
                                                     0.0f,
                                                     icon);
        const ImVec2 text_position{
                item_min.x + (toolbar_style.size.x - text_size.x) * 0.5f,
                item_min.y + (toolbar_style.size.y - text_size.y) * 0.5f
        };
        draw_list->AddText(font, toolbar_style.icon_size, text_position, ImGui::GetColorU32(text_color), icon);

        if (hovered) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(tooltip);
            ImGui::EndTooltip();
        }
        if (pressed && enabled) {
            on_pressed();
        }

        ImGui::PopID();
        if (!enabled) {
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
    };

    auto draw_operation_button = [&](const char* id, const char* icon, const char* tooltip, uint32_t operation) {
        draw_toolbar_button(id, icon, tooltip, node3d_editor->GetImGuizmoOperation() == operation, true, [&]() {
            node3d_editor->SetImGuizmoOperation(operation);
        });
    };

    auto draw_separator = [&]() {
        wrap_toolbar_item(9.0f);
        const ImVec2 cursor = ImGui::GetCursorScreenPos();
        auto* draw_list = ImGui::GetWindowDrawList();
        const float x = cursor.x + 4.0f;
        const float y = cursor.y + (button_size.y - separator_height) * 0.5f;
        draw_list->AddLine({x, y},
                           {x, y + separator_height},
                           ImGui::GetColorU32(ImGuiCol_Separator),
                           1.0f);
        ImGui::Dummy({9.0f, button_size.y});
        ImGui::SameLine();
    };

    draw_operation_button("select", ICON_MDI_CURSOR_DEFAULT, "Select", Node3DEditor::InvalidGuizmoOperation());
    draw_separator();
    draw_operation_button("translate", ICON_MDI_ARROW_ALL, "Translate", ImGuizmo::TRANSLATE);
    draw_operation_button("rotate", ICON_MDI_ROTATE_3D, "Rotate", ImGuizmo::ROTATE);
    draw_operation_button("scale", ICON_MDI_ARROW_EXPAND_ALL, "Scale", ImGuizmo::SCALE);
    draw_separator();
    draw_operation_button("universal", ICON_MDI_CROP_ROTATE, "Universal", ImGuizmo::UNIVERSAL);
    draw_separator();
    draw_operation_button("bounds", ICON_MDI_BORDER_NONE, "Bounds", ImGuizmo::BOUNDS);
    draw_separator();

    {
        const bool selected = node3d_editor->SnapGuizmo();
        draw_toolbar_button("snap", ICON_MDI_MAGNET, "Snap", selected, true, [&]() {
            node3d_editor->SnapGuizmo() = !selected;
        });
    }

    draw_separator();

    {
        const bool robot_available = active_robot != nullptr;
        const bool assembly_mode = robot_available && active_robot->GetMode() == RobotMode::Assembly;
        const bool motion_mode = robot_available && active_robot->GetMode() == RobotMode::Motion;

        draw_toolbar_button("assembly_mode", ICON_MDI_COGS, "Assembly Mode", assembly_mode, robot_available, [&]() {
            active_robot->SetMode(RobotMode::Assembly);
        });
        draw_toolbar_button("motion_mode", ICON_MDI_AXIS_ARROW, "Motion Mode", motion_mode, robot_available, [&]() {
            active_robot->SetMode(RobotMode::Motion);
        });
    }

    ImGui::PopStyleVar();
}

}
