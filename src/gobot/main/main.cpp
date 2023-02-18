/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#include "gobot/main/main.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/scene/scene_initializer.hpp"

namespace gobot {


static ProjectSettings* s_project_settings = nullptr;
static Input* s_input = nullptr;

bool Main::Setup() {
    s_project_settings = Object::New<ProjectSettings>();
    s_input = Object::New<Input>();

    return Setup2();
}

bool Main::Setup2() {
    SceneInitializer::Init();

    return true;
}


bool Main::Start() {
    MainLoop *main_loop = nullptr;
    main_loop = Object::New<SceneTree>();

    return true;
}

void Main::Cleanup() {
    SceneInitializer::Destroy();

    Object::Delete(s_input);
    Object::Delete(s_project_settings);
}

}