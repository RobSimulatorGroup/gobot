/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/


#include "gobot/editor/node3d_editor.hpp"
#include "gobot/error_macros.hpp"

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


void Node3DEditor::UpdateCameraByInput() {

}

}
