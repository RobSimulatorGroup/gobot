/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-28
 * SPDX-License-Identifier: Apache-2.0
 */


#include "gobot/editor/node3d_editor.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/os/os.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/scene_command.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/log.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_extension/gizmos/ImGuizmo.h"

#include <algorithm>
#include <cmath>

namespace gobot {

Node3DEditor* Node3DEditor::s_singleton = nullptr;

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

bool IsLockedByRobotMotionMode(Node* selected) {
    auto* robot = FindRobotAncestor(selected);
    return robot && robot != selected && robot->GetMode() == RobotMode::Motion;
}

bool ImGuiBlocksViewportInput() {
    ImGuiContext* context = ImGui::GetCurrentContext();
    if (context == nullptr) {
        return false;
    }

    return ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
}

bool IsFiniteVector(const Vector3& vector) {
    return std::isfinite(vector.x()) && std::isfinite(vector.y()) && std::isfinite(vector.z());
}

}

Node3DEditor::Node3DEditor() {
    s_singleton = this;
    imguizmo_operation_ = ImGuizmo::TRANSLATE;
    SetName("Node3DEditor");
    camera3d_ = Object::New<Camera3D>();
    camera3d_->SetName("EditorCamera");
    AddChild(camera3d_);
    ResetCamera();
    scene_view3d_panel_ = Object::New<SceneView3DPanel>();
    AddChild(scene_view3d_panel_);
}

void Node3DEditor::ResetCamera() {
    mouse_position_last_.x() = 0;
    mouse_position_last_.y() = 0;
    mouse_position_now_.x() = 0;
    mouse_position_now_.y() = 0;

    mouse_down_ = false;
    fly_speed_ = 3.0f;
    zoom_pivot_ = Vector3::Zero();
    zoom_pivot_valid_ = false;

    camera3d_->SetFovy(75.0);
    SetCameraOrbit({8.0f, 8.0f, 6.0f}, {0.0f, 0.0f, 0.0f}, Vector3::UnitZ());
}

EditorSceneViewState Node3DEditor::GetSceneViewState() const {
    if (camera3d_ == nullptr) {
        return {};
    }
    return EditorSceneViewState{
            .eye = camera3d_->GetViewMatrixEye(),
            .at = camera3d_->GetViewMatrixAt(),
            .up = camera3d_->GetViewMatrixUp(),
            .fov_y = camera3d_->GetFovy(),
    };
}

void Node3DEditor::ApplySceneViewState(const EditorSceneViewState& state) {
    const Vector3 view_direction = state.at - state.eye;
    if (!IsFiniteVector(state.eye) || !IsFiniteVector(state.at) ||
        !IsFiniteVector(state.up) || view_direction.isZero(CMP_EPSILON) ||
        state.up.isZero(CMP_EPSILON) || !std::isfinite(state.fov_y) ||
        state.fov_y <= 0.0 || state.fov_y >= 180.0) {
        ResetCamera();
        return;
    }

    Vector3 up = state.up.normalized();
    if (std::abs(view_direction.normalized().dot(up)) > 1.0 - CMP_EPSILON) {
        up = Vector3::UnitZ();
        if (std::abs(view_direction.normalized().dot(up)) > 1.0 - CMP_EPSILON) {
            up = Vector3::UnitY();
        }
    }

    camera3d_->SetFovy(state.fov_y);
    SetCameraOrbit(state.eye, state.at, up);
}


Node3DEditor::~Node3DEditor() {
    s_singleton = nullptr;
}


Node3DEditor* Node3DEditor::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize Node3DEditor");
    return s_singleton;
}

void Node3DEditor::NotificationCallBack(NotificationType notification) {
    switch (notification) {
        case NotificationType::Process: {
            auto delta = GetProcessDeltaTime();

            if (update_camera_) {
                UpdateCamera(delta);
            }
        }
    }
}

void Node3DEditor::SetNeedUpdateCamera(bool update_camera) {
    // If we are editing camera, let the editing finish itself.
    if (!editing_) {
        // Reset mouse scroll offset if we prepare to update camera
        if (!update_camera_ && update_camera) {
            Input::GetInstance()->SetScrollOffset(0.0);
        }
        update_camera_ = update_camera;
    }
}

void Node3DEditor::SetBlockCameraInput(bool block_camera_input) {
    block_camera_input_ = block_camera_input;
}

void Node3DEditor::SetViewportZoomPivot(const Vector3& pivot, bool valid) {
    zoom_pivot_ = pivot;
    zoom_pivot_valid_ = valid && IsFiniteVector(pivot);
}

void Node3DEditor::FocusNode(const Node3D* node) {
    if (node == nullptr || camera3d_ == nullptr || !node->IsInsideTree()) {
        return;
    }

    const Vector3 at = node->GetGlobalPosition();
    if (!IsFiniteVector(at)) {
        return;
    }

    Vector3 direction = camera3d_->GetViewMatrixAt() - camera3d_->GetViewMatrixEye();
    if (direction.squaredNorm() <= CMP_EPSILON2) {
        direction = Vector3{-1.0, -1.0, -0.75};
    }
    direction.normalize();

    Vector3 up = camera3d_->GetViewMatrixUp();
    if (!IsFiniteVector(up) || up.squaredNorm() <= CMP_EPSILON2 ||
        std::abs(up.normalized().dot(direction)) > 1.0 - CMP_EPSILON) {
        up = Vector3::UnitZ();
    } else {
        up.normalize();
    }

    const Vector3 eye = at - direction * std::max(distance_, 0.5f);
    SetCameraOrbit(eye, at, up);
}

void Node3DEditor::UpdateCamera(double delta_time) {
    auto* input = Input::GetInstance();
    if (block_camera_input_ || ImGuiBlocksViewportInput()) {
        mouse_down_ = false;
        editing_ = false;
        mouse_position_last_ = input->GetMousePosition();
        input->SetScrollOffset(0.0);
        return;
    }

    if (!mouse_down_) {
        mouse_position_last_ = input->GetMousePosition();
    }

    const float scroll_offset = input->GetScrollOffset();
    input->SetScrollOffset(0.0);

    const bool left_mouse_down = input->GetMouseClickedState(MouseButton::Left) == MouseClickedState::SingleClicked;
    const bool middle_mouse_down = input->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::SingleClicked;
    const bool right_mouse_down = input->GetMouseClickedState(MouseButton::Right) == MouseClickedState::SingleClicked;
    const bool shift_down = input->GetKeyHeld(KeyCode::LeftShift) ||
                            input->GetKeyHeld(KeyCode::RightShift);
    const bool ctrl_down = input->GetKeyHeld(KeyCode::LeftCtrl) ||
                           input->GetKeyHeld(KeyCode::RightCtrl);
    const bool alt_down = input->GetKeyHeld(KeyCode::LeftAlt) ||
                          input->GetKeyHeld(KeyCode::RightAlt);
    const bool gizmo_captures_mouse = ImGui::GetCurrentContext() != nullptr &&
                                      (ImGuizmo::IsUsing() || ImGuizmo::IsOver());
    const bool orbit_mouse_down = left_mouse_down && (alt_down || ctrl_down) && !gizmo_captures_mouse;
    const bool pan_mouse_down = (middle_mouse_down ||
                                 (left_mouse_down && shift_down && !ctrl_down && !alt_down)) &&
                                !gizmo_captures_mouse;
    const bool dolly_mouse_down = right_mouse_down && alt_down && !gizmo_captures_mouse;
    const bool fly_look_mouse_down = right_mouse_down && !alt_down && !gizmo_captures_mouse;

    mouse_down_ = orbit_mouse_down || pan_mouse_down || dolly_mouse_down || fly_look_mouse_down;

    Vector2i delta = Vector2i::Zero();
    if (mouse_down_) {
        mouse_position_now_ = input->GetMousePosition();
        delta = mouse_position_now_ - mouse_position_last_;

        delta[0] *= -1.0f;
        if (orbit_mouse_down || fly_look_mouse_down) {
            horizontal_angle_ += orbit_speed_ * static_cast<float>(delta[0]);
            vertical_angle_ -= orbit_speed_ * static_cast<float>(delta[1]);
            constexpr float kPitchLimit = static_cast<float>(M_PI * 0.5 - 0.01);
            vertical_angle_ = std::clamp(vertical_angle_, -kPitchLimit, kPitchLimit);
        }
        mouse_position_last_ = mouse_position_now_;

        editing_ = true;
    } else {
        editing_ = false;
    }

    const Vector3 direction = {
        std::cos(vertical_angle_) * std::cos(horizontal_angle_),
        std::cos(vertical_angle_) * std::sin(horizontal_angle_),
        std::sin(vertical_angle_)
    };

    Vector3 right = direction.cross(Vector3::UnitZ());
    if (right.isZero(CMP_EPSILON)) {
        right = Vector3::UnitX();
    } else {
        right.normalize();
    }

    auto eye = camera3d_->GetViewMatrixEye();
    auto at = camera3d_->GetViewMatrixAt();

    Vector3 up = right.cross(direction).normalized();
    if (input->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::DoubleClicked) {
        ResetCamera();
        return;
    } else if (pan_mouse_down) {
        const float scale = std::max(distance_, static_cast<float>(camera3d_->GetNear()) * 10.0f) * pan_speed_;
        const Vector3 offset = up * static_cast<float>(delta[1]) * scale +
                               right * static_cast<float>(delta[0]) * scale;
        eye += offset;
        at += offset;
    } else if (orbit_mouse_down) {
        eye = at - direction * distance_;
    } else if (dolly_mouse_down) {
        const float drag_delta = std::abs(delta[0]) >= std::abs(delta[1])
                ? -static_cast<float>(delta[0])
                : -static_cast<float>(delta[1]);
        const float amount = drag_delta *
                             std::max(distance_, static_cast<float>(camera3d_->GetNear()) * 10.0f) *
                             dolly_speed_;
        const Vector3 offset = direction * amount;
        eye += offset;
        at += offset;
    } else if (fly_look_mouse_down) {
        at = eye + direction * distance_;

        if (std::abs(scroll_offset) > CMP_EPSILON) {
            fly_speed_ *= std::pow(1.25f, scroll_offset);
            fly_speed_ = std::clamp(fly_speed_, 0.05f, 500.0f);
        }

        Vector3 movement = Vector3::Zero();
        movement += direction * static_cast<float>(input->GetKeyHeld(KeyCode::W) - input->GetKeyHeld(KeyCode::S));
        movement += right * static_cast<float>(input->GetKeyHeld(KeyCode::D) - input->GetKeyHeld(KeyCode::A));
        movement += Vector3::UnitZ() * static_cast<float>(input->GetKeyHeld(KeyCode::E) - input->GetKeyHeld(KeyCode::Q));
        if (movement.squaredNorm() > CMP_EPSILON2) {
            const float speed_scale = shift_down ? 4.0f : (ctrl_down ? 0.25f : 1.0f);
            const float frame_delta = static_cast<float>(std::clamp(delta_time, 0.0, 0.1));
            const Vector3 offset = movement.normalized() * fly_speed_ * speed_scale * frame_delta;
            eye += offset;
            at += offset;
        }
    } else {
        if (std::abs(scroll_offset) > CMP_EPSILON) {
            Vector3 dolly_direction = direction;
            float dolly_distance = distance_;
            if (zoom_pivot_valid_) {
                const Vector3 to_pivot = zoom_pivot_ - eye;
                if (to_pivot.squaredNorm() > CMP_EPSILON2) {
                    dolly_distance = static_cast<float>(to_pivot.norm());
                    dolly_direction = to_pivot / dolly_distance;
                }
            }
            const float amount = dolly_distance * (1.0f - std::pow(0.8f, scroll_offset));
            const Vector3 offset = dolly_direction * amount;
            eye += offset;
            at += offset;
        }
    }

    SetCameraOrbit(eye, at, up);
}

void Node3DEditor::SetCameraOrbit(const Vector3& eye, const Vector3& at, const Vector3& up) {
    const Vector3 direction = (at - eye).normalized();
    distance_ = std::max(static_cast<float>((at - eye).norm()), static_cast<float>(camera3d_->GetNear()));
    vertical_angle_ = static_cast<float>(std::asin(std::clamp(direction.z(), static_cast<RealType>(-1.0), static_cast<RealType>(1.0))));
    horizontal_angle_ = static_cast<float>(std::atan2(direction.y(), direction.x()));
    camera3d_->SetViewMatrix(eye, at, up);
}

void Node3DEditor::ApplyCameraViewMatrix(const Matrix4& view_matrix) {
    const Matrix4 inverse_view = view_matrix.inverse();
    const Vector3 eye = inverse_view.block<3, 1>(0, 3);
    const Vector3 forward = (-inverse_view.block<3, 1>(0, 2)).normalized();
    const Vector3 up = inverse_view.block<3, 1>(0, 1).normalized();
    const Vector3 at = eye + forward * distance_;

    SetCameraOrbit(eye, at, up);
}

void Node3DEditor::DrawViewManipulator(const ImVec2& position, const ImVec2& size) {
    Matrix4 view = camera3d_->GetViewMatrix();
    Matrix4 projection = camera3d_->GetProjectionMatrix();
    Matrix4 model = Matrix4::Identity();
    const Matrix4 original_view = view;

    ImGuizmo::ViewManipulate(view.data(), projection.data(), ImGuizmo::TRANSLATE, ImGuizmo::LOCAL,
                             model.data(), distance_, position, size, 0x10101010);
    if (!view.isApprox(original_view, CMP_EPSILON)) {
        ApplyCameraViewMatrix(view);
    }
}

void Node3DEditor::OnImGuizmo() {

    ImGuizmo::SetDrawlist();

    ImGuizmo::SetOrthographic(camera3d_->GetProjectionType() == Camera3D::ProjectionType::Orthogonal);

    auto window_width = (float)ImGui::GetWindowWidth();
    float view_manipulate_right = ImGui::GetWindowPos().x + window_width;
    float view_manipulate_top = ImGui::GetWindowPos().y;
    const ImVec2 view_manipulate_position{view_manipulate_right - 128, view_manipulate_top + 50};
    const ImVec2 view_manipulate_size{128, 128};

    if (Editor::GetInstance()->IsScenePlaySessionRunning()) {
        editing_ = false;
        DrawViewManipulator(view_manipulate_position, view_manipulate_size);
        return;
    }

    if (imguizmo_operation_ != InvalidGuizmoOperation()) {
        auto* selected = Editor::GetInstance()->GetSelected();
        auto* selected_node_3d = Object::PointerCastTo<Node3D>(selected);
        if (!selected_node_3d || !selected_node_3d->IsInsideTree()) {
            DrawViewManipulator(view_manipulate_position, view_manipulate_size);
            return;
        }
        if (IsLockedByRobotMotionMode(selected_node_3d)) {
            editing_ = false;
            DrawViewManipulator(view_manipulate_position, view_manipulate_size);
            return;
        }

        Matrix4 model_matrix = selected_node_3d->GetGlobalTransform().matrix();
        float object_matrix[16];
        for (int i = 0; i < 16; ++i) {
            object_matrix[i] = static_cast<float>(model_matrix.data()[i]);
        }

        Matrix4 view = camera3d_->GetViewMatrix();
        Matrix4 projection = camera3d_->GetProjectionMatrix();
        float snap[3] = {1.0f, 1.0f, 1.0f};
        if (imguizmo_operation_ == ImGuizmo::ROTATE) {
            snap[0] = 15.0f;
            snap[1] = 15.0f;
            snap[2] = 15.0f;
        } else if (imguizmo_operation_ == ImGuizmo::SCALE) {
            snap[0] = 0.1f;
            snap[1] = 0.1f;
            snap[2] = 0.1f;
        }

        bool changed = ImGuizmo::Manipulate(view.data(), projection.data(),
                                            static_cast<ImGuizmo::OPERATION>(imguizmo_operation_),
                                            ImGuizmo::LOCAL, object_matrix, nullptr,
                                            snap_guizmo_ ? snap : nullptr);

        if (changed) {
            for (int i = 0; i < 16; ++i) {
                model_matrix.data()[i] = static_cast<RealType>(object_matrix[i]);
            }
            auto* editor = Editor::GetInstance();
            if (auto* context = editor->GetEngineContext()) {
                context->ExecuteSceneCommand(std::make_unique<SetNode3DTransformCommand>(
                        selected_node_3d->GetInstanceId(),
                        Affine3(model_matrix),
                        true));
            }
        }

        editing_ = ImGuizmo::IsUsing();
    }

    DrawViewManipulator(view_manipulate_position, view_manipulate_size);

}

bool& Node3DEditor::SnapGuizmo() {
    return snap_guizmo_;
}


}

GOBOT_REGISTRATION {
    Class_<Node3DEditor>("Node3DEditor")
        .constructor()(CtorAsRawPtr);

};
