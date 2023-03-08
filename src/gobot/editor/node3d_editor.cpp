/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/


#include "gobot/editor/node3d_editor.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/os/os.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/log.hpp"

namespace gobot {

Node3DEditor* Node3DEditor::s_singleton = nullptr;

Node3DEditor::Node3DEditor() {
    s_singleton = this;
    camera3d_ = Object::New<Camera3D>();
    AddChild(camera3d_);
    ResetCamera();
}

void Node3DEditor::ResetCamera() {
    mouse_position_last_.x() = 0;
    mouse_position_last_.y() = 0;
    mouse_position_now_.x() = 0;
    mouse_position_now_.y() = 0;

    horizontal_angle_ = 0.01f;
    vertical_angle_ = 0.0f;

    eye_.x()  =   0.0f;
    eye_.y()  =   0.0f;
    eye_.z()  = -35.0f;
    at_.x()   =   0.0f;
    at_.y()   =   0.0f;
    at_.z()   =  -1.0f;
    up_.x()   =   0.0f;
    up_.y()   =   1.0f;
    up_.z()   =   0.0f;

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

            UpdateCamera(delta);
        }
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

    auto window = dynamic_cast<SceneTree*>(OS::GetInstance()->GetMainLoop())->GetRoot()->GetWindowsInterface();
    auto width = window->GetWidth();
    auto height = window->GetHeight();

    Vector2i delta;
    if (mouse_down_) {
        mouse_position_now_ = Input::GetInstance()->GetMousePosition();
        delta = mouse_position_now_ - mouse_position_last_;

        if (!Input::GetInstance()->GetKeyPressed(KeyCode::LeftShift)) {
            horizontal_angle_ += mouse_speed_ * float(delta[0]);
            vertical_angle_   -= mouse_speed_ * float(delta[1]);
        }
        mouse_position_last_ = mouse_position_now_;
    };

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

    up_ = right.cross(direction);
    if (Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::DoubleClicked) {
        ResetCamera();
    } else if (Input::GetInstance()->GetKeyPressed(KeyCode::LeftShift) &&
               Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::SingleClicked) {
        eye_ = eye_ + up_ * delta[1] * translation_speed_ + right * delta[0] * translation_speed_;
        at_ = eye_ + direction * distance_;
    } else if (Input::GetInstance()->GetMouseClickedState(MouseButton::Middle) == MouseClickedState::SingleClicked) {
        eye_ = at_ - direction * distance_;
    } else {
        eye_ = (direction * -1.0 * scroll_offset * delta_time * scroll_move_speed_) + eye_;
        at_ = eye_ + direction * distance_;
    }

    auto view = Matrix4::LookAt(eye_, at_, up_);
    camera3d_->SetGlobalTransform(Affine3(Matrix4::LookAt(eye_, at_, up_, Handedness::Right).matrix()));

    auto proj = Matrix4f::Perspective(camera3d_->GetFovy(), float(width)/float(height), 0.1f, 1000.0f);
    GET_RENDER_SERVER()->SetViewTransform(0, view, proj);
    GET_RENDER_SERVER()->SetViewRect(0, 0, 0, uint16_t(width), uint16_t(height));
}

}

GOBOT_REGISTRATION {
    Class_<Node3DEditor>("Node3DEditor")
        .constructor()(CtorAsRawPtr);

};

