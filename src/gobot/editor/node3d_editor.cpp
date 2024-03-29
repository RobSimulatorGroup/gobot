/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/


#include "gobot/editor/node3d_editor.hpp"
#include "gobot/editor/imgui/scene_view_3d_panel.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/os/os.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/log.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_extension/gizmos/ImGuizmo.h"
#include "glsl_shader_hpp/grid_frag.hpp"
#include "glsl_shader_hpp/grid_vert.hpp"

namespace gobot {

Node3DEditor* Node3DEditor::s_singleton = nullptr;

Node3DEditor::Node3DEditor() {
    s_singleton = this;
    camera3d_ = Object::New<Camera3D>();
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

    horizontal_angle_ = 0.01f;
    vertical_angle_ = 0.0f;

    // Set camera default position
    camera3d_->SetViewMatrix({0.0f, 0.0f, -20.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f});

    mouse_down_ = false;
    mouse_speed_ = 0.0020f;

    distance_ = 10;
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


void Node3DEditor::UpdateCamera(double delta_time) {
    if (!mouse_down_) {
        mouse_position_last_ = Input::GetInstance()->GetMousePosition();
    }

    auto scroll_offset = Input::GetInstance()->GetScrollOffset();
    Input::GetInstance()->SetScrollOffset(0.0);

    mouse_down_ = (Input::GetInstance()->GetMouseClickedState(MouseButton::Right) == MouseClickedState::SingleClicked) ||
                  (Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::SingleClicked);

    Vector2i delta;
    if (mouse_down_) {
        mouse_position_now_ = Input::GetInstance()->GetMousePosition();
        delta = mouse_position_now_ - mouse_position_last_;

        // TODO(wqq): Right hand or left hand
        delta[0] *= -1.0f;
        if (!Input::GetInstance()->GetKeyPressed(KeyCode::LeftShift)) {
            horizontal_angle_ += mouse_speed_ * float(delta[0]);
            vertical_angle_   -= mouse_speed_ * float(delta[1]);
        }
        mouse_position_last_ = mouse_position_now_;

        editing_ = true;
    } else {
        editing_ = false;
    }

    const Vector3 direction = {
        std::cos(vertical_angle_) * std::sin(horizontal_angle_),
        std::sin(vertical_angle_),
        std::cos(vertical_angle_) * std::cos(horizontal_angle_)
    };

    const Vector3 right = {
        std::sin(horizontal_angle_ - Math_HALF_PI),
        0.0f,
        std::cos(horizontal_angle_ - Math_HALF_PI),
    };

    auto up = camera3d_->GetViewMatrixUp();
    auto eye = camera3d_->GetViewMatrixEye();
    auto at = camera3d_->GetViewMatrixAt();

    up = right.cross(direction);
    if (Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::DoubleClicked) {
        ResetCamera();
    } else if (Input::GetInstance()->GetKeyPressed(KeyCode::LeftShift) &&
               Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::SingleClicked) {
        eye = eye + up * delta[1] * translation_speed_ + right * delta[0] * translation_speed_;
        at = eye + direction * distance_;
    } else if (Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::SingleClicked) {
        eye = at - direction * distance_;
    } else {
        eye = (direction * -1.0 * scroll_offset * delta_time * scroll_move_speed_) + eye;
        at = eye + direction * distance_;
    }

    camera3d_->SetViewMatrix(eye, at, up);
}


float objectMatrix[16] = {
                1.f, 0.f, 0.f, 0.f,
                0.f, 1.f, 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                0.f, 0.f, 0.f, 1.f };

void Node3DEditor::OnImGuizmo() {

    ImGuizmo::SetDrawlist();

    ImGuizmo::SetOrthographic(camera3d_->GetProjectionType() == Camera3D::ProjectionType::Orthogonal);

    auto window_width = (float)ImGui::GetWindowWidth();
    float view_manipulate_right = ImGui::GetWindowPos().x + window_width;
    float view_manipulate_top = ImGui::GetWindowPos().y;

    if (imguizmo_operation_ != InvalidGuizmoOperation()) {
        ImGuizmo::Manipulate(camera3d_->GetViewMatrix().data(), camera3d_->GetProjectionMatrix().data(),
                             static_cast<ImGuizmo::OPERATION>(imguizmo_operation_), ImGuizmo::LOCAL, objectMatrix);
    }

    ImGuizmo::ViewManipulate(camera3d_->GetViewMatrix().data(), camera3d_->GetViewMatrixEye().norm(),
                             ImVec2{view_manipulate_right - 128, view_manipulate_top + 50},
                             ImVec2(128, 128), 0x10101010);

}

bool& Node3DEditor::SnapGuizmo() {
    return snap_guizmo_;
}


}

GOBOT_REGISTRATION {
    Class_<Node3DEditor>("Node3DEditor")
        .constructor()(CtorAsRawPtr);

};

