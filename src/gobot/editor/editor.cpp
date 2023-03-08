/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-28
*/

#include "gobot/editor/editor.hpp"
#include "gobot/editor/node3d_editor.hpp"
#include "gobot/core/registration.hpp"

namespace gobot {

Editor* Editor::s_singleton = nullptr;

Editor::Editor() {
    s_singleton = this;

    node3d_editor_ = Object::New<Node3DEditor>();
    AddChild(node3d_editor_);
}

Editor::~Editor() {
    s_singleton = nullptr;
}

Editor* Editor::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize Editor");
    return s_singleton;
}

}

GOBOT_REGISTRATION {
    Class_<Editor>("Editor")
        .constructor()(CtorAsRawPtr);

};