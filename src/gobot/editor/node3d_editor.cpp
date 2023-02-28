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


void Node3DEditor::UpdateCamera(double interp_delta) {

    if (!mouse_down_)
    {
        mouse_position_last_ = Input::GetInstance()->GetMousePosition();
    }


    mouse_down_ = (bool)Input::GetInstance()->GetMouseClickedState(MouseButton::Right);

    if (mouse_down_)
    {
        mouse_position_now_ = Input::GetInstance()->GetMousePosition();
        const Vector2i delta = mouse_position_now_ - mouse_position_last_;

        angle_[0] += mouse_speed_ * float(delta[0]);
        angle_[1] += mouse_speed_ * float(delta[1]);

        mouse_position_last_ = mouse_position_now_;
    }

    const Vector3 direction = {
        std::cos(angle_[1]) * std::sin(angle_[0]),
        std::sin(angle_[1]),
        std::cos(angle_[1]) * std::cos(angle_[0])
    };

    const Vector3 right = {
        std::sin(angle_[0] - Math_PI * 0.5),
        0.0f,
        std::cos(angle_[0] - Math_PI * 0.5)
    };

    const Vector3 up = right.cross(direction);

    Vector3 m_eye = direction * (interp_delta * mouse_speed_) + camera3d_->GetPosition();
    Vector3 m_at = m_eye + direction;
    Vector3 m_up = right.cross(direction);


    auto window = dynamic_cast<SceneTree*>(OS::GetInstance()->GetMainLoop())->GetRoot()->GetWindowsInterface();
    auto width = window->GetWidth();
    auto height = window->GetHeight();

//    auto view = Matrix4f::LookAt({35.0f, 35.0f, -35.0f}, {0.0f, 0.0f, 0.0f});
//    auto proj = Matrix4f::Perspective(60.0, float(width)/float(height), 0.1f, 100.0f);
//    GET_RENDER_SERVER()->SetViewTransform(0, view, proj);
//    GET_RENDER_SERVER()->SetViewRect(0, 0, 0, uint16_t(width), uint16_t(height) );

    Affine3 view(Matrix4f::LookAt({35.0f, 35.0f, -35.0f}, {0.0f, 0.0f, 0.0f}, m_up));
    camera3d_->SetTransform(view);
    auto proj = Matrix4f::Perspective(60.0, float(width)/float(height), 0.1f, 100.0f);
    GET_RENDER_SERVER()->SetViewTransform(0, view.matrix(), proj);
    GET_RENDER_SERVER()->SetViewRect(0, 0, 0, uint16_t(width), uint16_t(height));
}

}

GOBOT_REGISTRATION {
    Class_<Node3DEditor>("Node3DEditor")
        .constructor()(CtorAsRawPtr);

};

