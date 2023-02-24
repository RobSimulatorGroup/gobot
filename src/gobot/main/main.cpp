/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#include "gobot/main/main.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/config/engine.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/core/os/os.hpp"
#include <cxxopts.hpp>
#include <bgfx/bgfx.h>
#include "gobot/scene/window.hpp"

namespace gobot {

static Engine *s_engine = nullptr;
static ProjectSettings* s_project_settings = nullptr;
static Input* s_input = nullptr;
static RenderServer* s_render_server = nullptr;

Main::TimePoint Main::s_last_ticks = std::chrono::high_resolution_clock::now();

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


    s_engine = Object::New<Engine>();
    s_project_settings = Object::New<ProjectSettings>();
    s_input = Object::New<Input>();

    if (result.count("path")) {
        if (s_project_settings->SetProjectPath(result["path"].as<std::string>().c_str())) {
            return false;
        }
    }


    return Setup2();
}

bool Main::Setup2() {
    s_render_server = Object::New<RenderServer>();


    SceneInitializer::Init();

    return true;
}


bool Main::Start() {
    auto* main_loop = Object::New<SceneTree>();
    OS::GetInstance()->SetMainLoop(main_loop);

    s_render_server->InitWindow();
    s_render_server->SetDebug(RenderDebugFlags::DebugTextDisplay);

    return true;
}


bool Main::Iteration()
{

    auto window = dynamic_cast<SceneTree*>(OS::GetInstance()->GetMainLoop())->GetRoot()->GetWindowsInterface();

    auto time_now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::ratio<1>>(time_now-s_last_ticks).count();

    s_last_ticks = time_now;

    bool exit = false;
    if (OS::GetInstance()->GetMainLoop()->PhysicsProcess(duration)) {
        exit = true;
    }

    if (OS::GetInstance()->GetMainLoop()->Process(duration)) {
        exit = true;
    }

    // Set view 0 default viewport.
    auto width = window->GetWidth();
    auto height = window->GetHeight();

    GET_RENDER_SERVER()->SetViewRect(0, 0, 0, uint16_t(width), uint16_t(height) );

    // This dummy draw call is here to make sure that view 0 is cleared
    // if no other draw calls are submitted to view 0.
    GET_RENDER_SERVER()->Touch(0);

    // Use debug font to print information about this example.
    GET_RENDER_SERVER()->DebugTextClear();

    GET_RENDER_SERVER()->DebugTextPrintf(0, 1, 0x0f, "Color can be changed with ANSI \x1b[9;me\x1b[10;ms\x1b[11;mc\x1b[12;ma\x1b[13;mp\x1b[14;me\x1b[0m code too.");

    GET_RENDER_SERVER()->DebugTextPrintf(80, 1, 0x0f, "\x1b[;0m    \x1b[;1m    \x1b[; 2m    \x1b[; 3m    \x1b[; 4m    \x1b[; 5m    \x1b[; 6m    \x1b[; 7m    \x1b[0m");
    GET_RENDER_SERVER()->DebugTextPrintf(80, 2, 0x0f, "\x1b[;8m    \x1b[;9m    \x1b[;10m    \x1b[;11m    \x1b[;12m    \x1b[;13m    \x1b[;14m    \x1b[;15m    \x1b[0m");

    const RenderStats* stats = GET_RENDER_SERVER()->GetStats();
    GET_RENDER_SERVER()->DebugTextPrintf(0, 2, 0x0f, "Backbuffer %dW x %dH in pixels, debug text %dW x %dH in characters."
            , stats->width
            , stats->height
            , stats->textWidth
            , stats->textHeight
    );

    // Advance to next frame. Rendering thread will be kicked to
    // process submitted rendering primitives.
    GET_RENDER_SERVER()->Frame();

    return exit;

}

void Main::Cleanup() {
    SceneInitializer::Destroy();

    Object::Delete(s_input);
    Object::Delete(s_project_settings);

    bgfx::shutdown();
}

}