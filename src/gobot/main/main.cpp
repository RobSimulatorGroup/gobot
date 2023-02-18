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
#include "gobot/core/os/os.hpp"
#include <cxxopts.hpp>

namespace gobot {


static ProjectSettings* s_project_settings = nullptr;
static Input* s_input = nullptr;

bool Main::Setup(int argc, char** argv) {
    cxxopts::Options options("gobot_editor",
                             R"(
The gobot is a robot simulation platform.
Free and open source software under the terms of the LGPL3 license.
Copyright(c) 2021-2023, RobSimulatorGroup)");

    options.add_options()
            ("path", "gobot project path", cxxopts::value<std::string>())
            ("version", "query version of gobot")
            ("v,verbose", "verbose output", cxxopts::value<bool>())
            ("q,quiet", "quieter output", cxxopts::value<bool>())
            ("h,help", "Print usage")
            ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }


    s_project_settings = Object::New<ProjectSettings>();
    s_input = Object::New<Input>();


    return Setup2();
}

bool Main::Setup2() {
    SceneInitializer::Init();

    return true;
}


bool Main::Start() {
    MainLoop* main_loop = Object::New<SceneTree>();
    OS::GetInstance()->SetMainLoop(main_loop);

    return true;
}

bool Main::Iteration()
{
    bool exit = false;
    if (OS::GetInstance()->GetMainLoop()->PhysicsProcess(0.01)) {
        exit = true;
    }

    if (OS::GetInstance()->GetMainLoop()->Process(0.01)) {
        exit = true;
    }

    return exit;

}

void Main::Cleanup() {
    SceneInitializer::Destroy();

    Object::Delete(s_input);
    Object::Delete(s_project_settings);
}

}