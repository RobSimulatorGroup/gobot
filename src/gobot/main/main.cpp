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
#include "gobot/scene/window.hpp"
#include "gobot/rendering/load_shader.hpp"
#include "gobot/rendering/debug_draw/debug_draw.hpp"
#include "gobot/core/math/geometry.hpp"
#include <cxxopts.hpp>
#include <bgfx/bgfx.h>

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

struct PosColorVertex
{
    float m_x;
    float m_y;
    float m_z;
    uint32_t m_abgr;

    static void init()
    {
        ms_layout
                .begin()
                .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
                .add(bgfx::Attrib::Color0,   4, bgfx::AttribType::Uint8, true)
                .end();
    };

    static bgfx::VertexLayout ms_layout;
};

bgfx::VertexLayout PosColorVertex::ms_layout;

static PosColorVertex s_cubeVertices[] =
        {
                {-5.0f,  5.0f,  5.0f, 0xff000000 },
                { 5.0f,  5.0f,  5.0f, 0xff0000ff },
                {-5.0f, -5.0f,  5.0f, 0xff00ff00 },
                { 5.0f, -5.0f,  5.0f, 0xff00ffff },
                {-5.0f,  5.0f, -5.0f, 0xffff0000 },
                { 5.0f,  5.0f, -5.0f, 0xffff00ff },
                {-5.0f, -5.0f, -5.0f, 0xffffff00 },
                { 5.0f, -5.0f, -5.0f, 0xffffffff },
        };

static const uint16_t s_cubeTriList[] =
        {
                0, 1, 2, // 0
                1, 3, 2,
                4, 6, 5, // 2
                5, 6, 7,
                0, 2, 4, // 4
                4, 2, 6,
                1, 5, 3, // 6
                5, 7, 3,
                0, 4, 1, // 8
                4, 5, 1,
                2, 3, 6, // 10
                6, 3, 7,
        };

static const uint16_t s_cubeTriStrip[] =
        {
                0, 1, 2,
                3,
                7,
                1,
                5,
                0,
                4,
                2,
                6,
                7,
                4,
                5,
        };

static const uint16_t s_cubeLineList[] =
        {
                0, 1,
                0, 2,
                0, 4,
                1, 3,
                1, 5,
                2, 3,
                2, 6,
                3, 7,
                4, 5,
                4, 6,
                5, 7,
                6, 7,
        };

static const uint16_t s_cubeLineStrip[] =
        {
                0, 2, 3, 1, 5, 7, 6, 4,
                0, 2, 6, 4, 5, 7, 3, 1,
                0,
        };

static const uint16_t s_cubePoints[] =
        {
                0, 1, 2, 3, 4, 5, 6, 7
        };

static const char* s_ptNames[]
        {
                "Triangle List",
                "Triangle Strip",
                "Lines",
                "Line Strip",
                "Points",
        };

static const uint64_t s_ptState[]
        {
                UINT64_C(0),
                BGFX_STATE_PT_TRISTRIP,
                BGFX_STATE_PT_LINES,
                BGFX_STATE_PT_LINESTRIP,
                BGFX_STATE_PT_POINTS,
        };


bgfx::VertexBufferHandle m_vbh;
bgfx::IndexBufferHandle m_ibh[1];
bgfx::ProgramHandle m_program;

bool Main::Start() {
    auto* main_loop = Object::New<SceneTree>();
    OS::GetInstance()->SetMainLoop(main_loop);

    s_render_server->InitWindow();
    USING_ENUM_BITWISE_OPERATORS;
    s_render_server->SetDebug(RenderDebugFlags::DebugTextDisplay);
    s_render_server->SetViewClear(0, RenderClearFlags::Depth | RenderClearFlags::Color);


    PosColorVertex::init();
    // Create static vertex buffer.
    m_vbh = bgfx::createVertexBuffer(
            // Static data can be passed with bgfx::makeRef
            bgfx::makeRef(s_cubeVertices, sizeof(s_cubeVertices) )
            , PosColorVertex::ms_layout
    );

    // Create static index buffer for triangle list rendering.
    m_ibh[0] = bgfx::createIndexBuffer(
            // Static data can be passed with bgfx::makeRef
            bgfx::makeRef(s_cubeTriList, sizeof(s_cubeTriList) )
    );

    // Create program from shaders.
    ShaderHandle vsh = LoadShader("vs_cubes");
    ShaderHandle fsh = LoadShader("fs_cubes");
    m_program = s_render_server->CreateProgram(vsh, fsh, true);

    ddInit();

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

    auto view = Matrix4f::LookAt({35.0f, 35.0f, -35.0f}, {0.0f, 0.0f, 0.0f});
    auto proj = Matrix4f::Perspective(60.0, float(width)/float(height), 0.1f, 100.0f);
    GET_RENDER_SERVER()->SetViewTransform(0, view, proj);
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

//     Set vertex and index buffer.
    GET_RENDER_SERVER()->SetVertexBuffer(0, m_vbh);
    GET_RENDER_SERVER()->SetIndexBuffer(m_ibh[0]);

    // Set render states.
    USING_ENUM_BITWISE_OPERATORS;
    GET_RENDER_SERVER()->SetState(RenderStateFlags::Default);

    static int i = 0;
    Isometry3f isometry_3_f = Isometry3f::Identity();
    isometry_3_f.SetEulerAngle({0.01 * i++,
                                -0.01 * i, 0}, EulerOrder::RXYZ);

    GET_RENDER_SERVER()->SetTransform(isometry_3_f.matrix());

    // Submit primitive for rendering to view 0.
    GET_RENDER_SERVER()->Submit(0, m_program);

    // Advance to next frame. Rendering thread will be kicked to
    // process submitted rendering primitives.

    DebugDrawEncoder dde;

    dde.begin(0);
    dde.drawAxis(0.0f, 0.0f, 0.0f, 1000000.0);

    dde.push();
    bx::Aabb aabb =
            {
                    {  5.0f, 1.0f, 1.0f },
                    { 10.0f, 5.0f, 5.0f },
            };
    dde.setWireframe(true);
//    dde.setColor(intersect(&dde, ray, aabb) ? kSelected : 0xff00ff00);
    dde.draw(aabb);
    dde.pop();


    {
        const bx::Vec3 normal = { 0.0f,  1.0f, 0.0f };
        const bx::Vec3 pos    = { 0.0f, -2.0f, 0.0f };

        bx::Plane plane(bx::init::None);
        bx::calcPlane(plane, normal, pos);

        dde.drawGrid(Axis::Y, pos, 128, 1.0f);
    }

    dde.end();

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