/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/


#include "gobot/editor/node3d_editor.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/os/os.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/robot_3d.hpp"
#include "gobot/log.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_extension/gizmos/ImGuizmo.h"
#include "glsl_shader_hpp/grid_frag.hpp"
#include "glsl_shader_hpp/grid_vert.hpp"

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

    shader_material_ = MakeRef<ShaderMaterial>();

    auto vs_shader = MakeRef<Shader>();
    vs_shader->SetShaderType(ShaderType::VertexShader);
    vs_shader->SetCode(GRID_VERT);
    auto fs_shader = MakeRef<Shader>();
    fs_shader->SetShaderType(ShaderType::FragmentShader);
    fs_shader->SetCode(GRID_FRAG);
    auto shader_program = MakeRef<RasterizerShaderProgram>();
    shader_program->SetRasterizerShader(vs_shader, fs_shader);

    shader_material_->SetShaderProgram(shader_program);
}

void Node3DEditor::ResetCamera() {
    mouse_position_last_.x() = 0;
    mouse_position_last_.y() = 0;
    mouse_position_now_.x() = 0;
    mouse_position_now_.y() = 0;

    mouse_down_ = false;
    mouse_speed_ = 0.0020f;

    SetCameraOrbit({8.0f, 8.0f, 6.0f}, {0.0f, 0.0f, 0.0f}, Vector3::UnitZ());
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

void Node3DEditor::UpdateCamera(double delta_time) {
    if (block_camera_input_ || ImGuiBlocksViewportInput()) {
        mouse_down_ = false;
        editing_ = false;
        mouse_position_last_ = Input::GetInstance()->GetMousePosition();
        Input::GetInstance()->SetScrollOffset(0.0);
        return;
    }

    if (!mouse_down_) {
        mouse_position_last_ = Input::GetInstance()->GetMousePosition();
    }

    auto scroll_offset = Input::GetInstance()->GetScrollOffset();
    Input::GetInstance()->SetScrollOffset(0.0);

    const bool left_mouse_down = Input::GetInstance()->GetMouseClickedState(MouseButton::Left) == MouseClickedState::SingleClicked;
    const bool middle_mouse_down = Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::SingleClicked;
    const bool right_mouse_down = Input::GetInstance()->GetMouseClickedState(MouseButton::Right) == MouseClickedState::SingleClicked;
    const bool shift_down = Input::GetInstance()->GetKeyPressed(KeyCode::LeftShift) ||
                            Input::GetInstance()->GetKeyPressed(KeyCode::RightShift);
    const bool ctrl_down = Input::GetInstance()->GetKeyPressed(KeyCode::LeftCtrl) ||
                           Input::GetInstance()->GetKeyPressed(KeyCode::RightCtrl);
    const bool gizmo_captures_mouse = ImGuizmo::IsUsing() || ImGuizmo::IsOver();
    const bool orbit_mouse_down = (right_mouse_down || (left_mouse_down && ctrl_down)) && !gizmo_captures_mouse;
    const bool pan_mouse_down = middle_mouse_down || (left_mouse_down && shift_down && !ctrl_down && !gizmo_captures_mouse);

    mouse_down_ = orbit_mouse_down || pan_mouse_down;

    Vector2i delta;
    if (mouse_down_) {
        mouse_position_now_ = Input::GetInstance()->GetMousePosition();
        delta = mouse_position_now_ - mouse_position_last_;

        // TODO(wqq): Right hand or left hand
        delta[0] *= -1.0f;
        if (orbit_mouse_down) {
            horizontal_angle_ += mouse_speed_ * float(delta[0]);
            vertical_angle_   -= mouse_speed_ * float(delta[1]);
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
    if (Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::DoubleClicked) {
        ResetCamera();
        return;
    } else if (pan_mouse_down) {
        const Vector3 offset = up * delta[1] * translation_speed_ + right * delta[0] * translation_speed_;
        eye += offset;
        at += offset;
    } else if (orbit_mouse_down) {
        eye = at - direction * distance_;
    } else {
        if (std::abs(scroll_offset) > CMP_EPSILON) {
            distance_ *= std::pow(0.85f, scroll_offset);
            distance_ = std::max(distance_, 0.1f);
        }
        eye = at - direction * distance_;
    }

    SetCameraOrbit(eye, at, up);
}

void Node3DEditor::SetCameraOrbit(const Vector3& eye, const Vector3& at, const Vector3& up) {
    const Vector3 direction = (at - eye).normalized();
    distance_ = std::max(static_cast<float>((at - eye).norm()), 0.1f);
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
            selected_node_3d->SetGlobalTransform(Affine3(model_matrix));
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
