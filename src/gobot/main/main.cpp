/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-10
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/main/main.hpp"
#include "gobot/core/profile.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/config/engine.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/core/io/variant_serializer.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/drivers/opengl/rasterizer_gl.hpp"
#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/simulation/simulation_server.hpp"
#include "gobot/core/os/os.hpp"
#include "gobot/python/python_script_runner.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/core/math/geometry.hpp"
#include "cxxopts.hpp"
#include "imgui.h"

#include <fstream>
#include <optional>
#include <thread>
#include <vector>

namespace gobot {

static Engine *s_engine = nullptr;
static ProjectSettings* s_project_settings = nullptr;
static Input* s_input = nullptr;
static RenderServer* s_render_server = nullptr;
static PhysicsServer* s_physics_server = nullptr;
static SimulationServer* s_simulation_server = nullptr;

namespace {
using BenchmarkClock = std::chrono::steady_clock;
bool s_worker_scheduling = true;
double s_render_interval = 1.0 / 60.0;
auto s_next_render = BenchmarkClock::now();
auto s_previous_render = BenchmarkClock::now();
double s_dispatch_since_render_ms = 0;
struct BenchmarkSample {
    double frame_ms;
    double physics_dispatch_ms;
    double process_ms;
    double draw_ms;
    double simulation_time;
    std::uint64_t simulation_tick;
    RenderResourceStats resources;
};
struct BenchmarkPhysicsSample {
    std::size_t render_frame;
    std::uint64_t tick;
    int dispatch_ticks;
    double step_ms;
    double solve_ms;
    int newton_iterations;
    int line_search_iterations;
};
struct EditorBenchmark {
    std::string output;
    std::size_t warmup{100};
    std::size_t frames{0};
    bool play{false};
    bool started{false};
    bool written{false};
    BenchmarkClock::time_point launch{BenchmarkClock::now()};
    double scene_ready_ms{0.0};
    double first_play_ms{0.0};
    std::optional<double> world_ready_ms;
    BenchmarkClock::time_point play_requested;
    Vector2i window_size{0, 0};
    std::vector<BenchmarkSample> samples;
    std::vector<BenchmarkPhysicsSample> physics_samples;
    std::uint64_t last_physics_tick = 0;
} s_benchmark;

double Milliseconds(BenchmarkClock::time_point begin, BenchmarkClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void WriteEditorBenchmark() {
    if (s_benchmark.output.empty() || s_benchmark.written) {
        return;
    }
    Json samples = Json::array();
    for (const auto& sample : s_benchmark.samples) {
        samples.push_back({sample.frame_ms, sample.physics_dispatch_ms, sample.process_ms,
                           sample.draw_ms, sample.simulation_time, sample.simulation_tick,
                           sample.resources.mesh_entries, sample.resources.texture_entries,
                           sample.resources.resident_bytes, sample.resources.uploaded_bytes,
                           sample.resources.geometry_uploads, sample.resources.index_uploads,
                           sample.resources.image_uploads, sample.resources.upload_ms});
    }
    Json physics_samples = Json::array();
    for (const auto& sample : s_benchmark.physics_samples) {
        physics_samples.push_back({sample.render_frame, sample.tick, sample.dispatch_ticks,
                sample.step_ms, sample.solve_ms, sample.newton_iterations, sample.line_search_iterations});
    }
    Json report = {
        {"schema", 1}, {"commit", Engine::GetBuildCommit()},
        {"project", s_project_settings->GetProjectPath()},
        {"play", s_benchmark.play}, {"warmup_frames", s_benchmark.warmup},
        {"requested_frames", s_benchmark.frames},
        {"scene_ready_ms", s_benchmark.scene_ready_ms},
        {"first_play_ms", s_benchmark.first_play_ms},
        {"world_ready_ms", s_benchmark.world_ready_ms ? Json(*s_benchmark.world_ready_ms) : Json(nullptr)},
        {"world_ready", s_simulation_server->IsSessionReady()},
        {"window_size", {s_benchmark.window_size.x(), s_benchmark.window_size.y()}},
        {"fixed_dt", s_simulation_server->GetFixedTimeStep()},
        {"physics_settings", VariantSerializer::VariantToJson(s_simulation_server->GetPhysicsWorldSettings())},
        {"time_scale", s_simulation_server->GetTimeScale()},
        {"max_sub_steps", s_simulation_server->GetMaxSubSteps()},
        {"scheduling", s_simulation_server->IsSessionAsynchronous() ? "worker" : "synchronous"},
        {"requested_scheduling", s_worker_scheduling ? "worker" : "synchronous"},
        {"render_fps_limit", 1.0 / s_render_interval},
        {"physics_backend", static_cast<int>(s_simulation_server->GetBackendType())},
        {"renderer", s_render_server->GetSceneRendererStats().active_mode == SceneRendererMode::Raster
                ? "OpenGL" : s_render_server->GetSceneRendererCapabilities().backend_name},
        {"renderer_mode", static_cast<int>(s_render_server->GetSceneRendererStats().active_mode)},
        {"faulted", s_simulation_server->IsFaulted()},
        {"error", s_simulation_server->GetLastError()},
        {"columns", {"frame_ms", "physics_dispatch_ms", "process_ms", "draw_ms",
                     "simulation_time", "simulation_tick", "mesh_cache_entries", "texture_cache_entries",
                     "render_cache_bytes", "uploaded_bytes_total", "geometry_uploads_total", "index_uploads_total",
                     "image_uploads_total", "upload_ms_total"}},
        {"samples", std::move(samples)},
        {"physics_columns", {"render_frame", "tick", "dispatch_ticks", "step_ms", "solve_ms",
                             "newton_iterations", "line_search_iterations"}},
        {"physics_samples", std::move(physics_samples)},
    };
    std::ofstream output(s_benchmark.output);
    output << report.dump(2) << '\n';
    if (!output) {
        LOG_ERROR("Cannot write editor benchmark to '{}'.", s_benchmark.output);
    } else {
        s_benchmark.written = true;
    }
}
} // namespace

Main::TimePoint Main::s_last_ticks = std::chrono::high_resolution_clock::now();

bool Main::Setup(int argc, char** argv) {
    cxxopts::Options options("gobot_editor",
                             R"(
The gobot is a robot simulation platform.
Free and open source software under the terms of the Apache-2.0 license.
Copyright(c) 2021-2026, RobSimulatorGroup)");

    options.add_options()
            ("path", "gobot project path", cxxopts::value<std::string>())
            ("version", "query version of gobot")
            ("v,verbose", "verbose output", cxxopts::value<bool>())
            ("q,quiet", "quieter output", cxxopts::value<bool>())
            ("benchmark-output", "Write bounded editor frame samples to JSON", cxxopts::value<std::string>())
            ("benchmark-frames", "Measured frames after warmup", cxxopts::value<std::size_t>()->default_value("500"))
            ("benchmark-warmup", "Warmup frames", cxxopts::value<std::size_t>()->default_value("100"))
            ("benchmark-play", "Start scene Play before frame capture", cxxopts::value<bool>()->default_value("false"))
            ("physics-scheduling", "Editor physics scheduling: worker or synchronous", cxxopts::value<std::string>()->default_value("worker"))
            ("render-fps", "Editor presentation limit, independent of physics Hz", cxxopts::value<int>()->default_value("60"))
            ("h,help", "Print usage")
            ;

    auto result = options.parse(argc, argv);

    const auto scheduling = result["physics-scheduling"].as<std::string>();
    const int render_fps = result["render-fps"].as<int>();
    if ((scheduling != "worker" && scheduling != "synchronous") || render_fps < 1 || render_fps > 1000) {
        std::cerr << "Expected worker/synchronous scheduling and render-fps in [1, 1000].\n";
        return false;
    }
    s_worker_scheduling = scheduling == "worker";
    s_render_interval = 1.0 / render_fps;

    if (result.count("benchmark-output")) {
        s_benchmark.output = result["benchmark-output"].as<std::string>();
        s_benchmark.frames = result["benchmark-frames"].as<std::size_t>();
        s_benchmark.warmup = result["benchmark-warmup"].as<std::size_t>();
        s_benchmark.play = result["benchmark-play"].as<bool>();
        if (s_benchmark.output.empty() || s_benchmark.frames == 0 ||
            s_benchmark.frames > 1000000 || s_benchmark.warmup > 1000000) {
            std::cerr << "Invalid benchmark output or frame count (maximum 1000000).\n";
            return false;
        }
        s_benchmark.samples.reserve(s_benchmark.frames + s_benchmark.warmup);
    }

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("version")) {
        std::cout << "gobot " << Engine::GetVersionString();
        const std::string commit = Engine::GetBuildCommit();
        if (!commit.empty()) {
            std::cout << " (" << commit << ")";
        }
        std::cout << std::endl;
        exit(0);
    }


    s_engine = Object::New<Engine>();
    s_project_settings = Object::New<ProjectSettings>();
    s_input = Object::New<Input>();
    s_physics_server = Object::New<PhysicsServer>();
    s_simulation_server = Object::New<SimulationServer>();
    s_simulation_server->SetAsyncSteppingEnabled(s_worker_scheduling);

    if (result.count("path")) {
        if (!s_project_settings->SetProjectPath(result["path"].as<std::string>().c_str())) {
            return false;
        }
    }


    return Setup2();
}

bool Main::Setup2() {
    Window::SetWindowFactory([]() -> std::unique_ptr<WindowInterface> {
        return std::make_unique<SDLWindow>();
    });
    opengl::GLRasterizer::MakeCurrent();
    s_render_server = Object::New<RenderServer>();

    SceneInitializer::Init();

    return true;
}

bool Main::Start() {
    auto* main_loop = Object::New<SceneTree>();
    main_loop->SetPhysicsProcessDriver([](double delta, const SceneTree::PhysicsNotification& notify) {
        if (s_simulation_server->HasActiveSession()) {
            s_simulation_server->AdvanceRealtime(static_cast<RealType>(delta), notify);
        } else {
            notify(delta);
        }
    });
    if (s_worker_scheduling && !main_loop->GetRoot()->GetWindow()->SetVSyncEnabled(false)) {
        LOG_WARN("Could not disable blocking presentation; physics callback cadence may be limited by display refresh.");
    }
    if (!s_benchmark.output.empty()) {
        main_loop->GetRoot()->GetWindow()->Restore();
        main_loop->GetRoot()->GetWindow()->SetWindowSize(1920, 1080);
        s_benchmark.window_size = main_loop->GetRoot()->GetWindow()->GetWindowSize();
    }

    USING_ENUM_BITWISE_OPERATORS;

    auto* editor = Object::New<Editor>();
    main_loop->GetRoot()->AddChild(editor);

    OS::GetInstance()->SetMainLoop(main_loop);
    s_previous_render = s_next_render = BenchmarkClock::now();

    return true;
}


bool Main::Iteration()
{
    GOBOT_PROFILE_ZONE("Main::Iteration");
    if (!s_benchmark.output.empty() && !s_benchmark.started) {
        auto* editor = Editor::GetInstanceOrNull();
        if (editor != nullptr && editor->HasCurrentScenePath()) {
            auto* tree = static_cast<SceneTree*>(OS::GetInstance()->GetMainLoop());
            s_benchmark.window_size = tree->GetRoot()->GetWindow()->GetWindowSize();
            const auto ready = BenchmarkClock::now();
            s_benchmark.scene_ready_ms = Milliseconds(s_benchmark.launch, ready);
            s_benchmark.play_requested = ready;
            if (s_benchmark.play && !editor->PlayScene()) {
                LOG_ERROR("Benchmark Play failed: {}", editor->GetScenePlaySessionLastError());
                WriteEditorBenchmark();
                return true;
            }
            s_benchmark.first_play_ms = Milliseconds(ready, BenchmarkClock::now());
            s_benchmark.started = true;
        }
    }
    auto time_now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::ratio<1>>(time_now - s_last_ticks).count();
    s_last_ticks = time_now;

    const auto now = BenchmarkClock::now();
    const bool render_due = !s_worker_scheduling || now >= s_next_render;
    const double render_delta = std::chrono::duration<double>(now - s_previous_render).count();

    bool exit = false;
    const auto physics_begin = BenchmarkClock::now();
    {
        GOBOT_PROFILE_ZONE("Main::PhysicsProcess");
        const bool runtime_worker = s_simulation_server->IsSessionAsynchronous();
        if ((runtime_worker || render_due) &&
            OS::GetInstance()->GetMainLoop()->PhysicsProcess(runtime_worker ? duration : render_delta)) {
            exit = true;
        }
    }

    const auto process_begin = BenchmarkClock::now();
    if (s_benchmark.started && s_benchmark.play && !s_benchmark.world_ready_ms &&
        s_simulation_server->IsSessionReady()) {
        s_benchmark.world_ready_ms = Milliseconds(s_benchmark.play_requested, process_begin);
    }
    if (s_benchmark.started && s_simulation_server->GetLastStepCount() > 0 &&
        s_simulation_server->GetFrameCount() != s_benchmark.last_physics_tick) {
        const auto& diagnostics = s_simulation_server->GetLastPhysicsStepResult().diagnostics;
        s_benchmark.last_physics_tick = s_simulation_server->GetFrameCount();
        s_benchmark.physics_samples.push_back({s_benchmark.samples.size(), s_benchmark.last_physics_tick,
                s_simulation_server->GetLastStepCount(), diagnostics.total_step_time_seconds * 1000,
                diagnostics.solve_time_seconds * 1000, diagnostics.newton_iterations, diagnostics.line_search_iterations});
    }
    if (s_benchmark.started && s_simulation_server->IsFaulted()) exit = true;
    s_dispatch_since_render_ms += Milliseconds(physics_begin, process_begin);
    if (!render_due) {
        if (!exit) std::this_thread::sleep_for(std::chrono::microseconds(250));
        if (exit) WriteEditorBenchmark();
        return exit;
    }
    s_previous_render = now;
    const auto interval = std::chrono::duration_cast<BenchmarkClock::duration>(std::chrono::duration<double>(s_render_interval));
    s_next_render += interval;
    if (s_next_render <= now) s_next_render = now + interval;
    GOBOT_PROFILE_PLOT("frame_ms", render_delta * 1000.0);
    if (render_delta > 0) GOBOT_PROFILE_PLOT("render_fps", 1.0 / render_delta);

    {
        GOBOT_PROFILE_ZONE("Main::Process");
        if (OS::GetInstance()->GetMainLoop()->Process(render_delta)) {
            exit = true;
        }
    }

    const auto draw_begin = BenchmarkClock::now();

    {
        GOBOT_PROFILE_ZONE("Main::Draw");
        RS::GetInstance()->Draw();
    }

    if (s_benchmark.started && (!s_benchmark.play || s_benchmark.world_ready_ms)) {
        const auto end = BenchmarkClock::now();
        s_benchmark.samples.push_back({render_delta * 1000.0,
            s_dispatch_since_render_ms, Milliseconds(process_begin, draw_begin),
            Milliseconds(draw_begin, end), s_simulation_server->GetSimulationTime(),
            s_simulation_server->GetFrameCount(), s_render_server->GetSceneRendererStats().resources});
        exit = exit || s_benchmark.samples.size() >= s_benchmark.warmup + s_benchmark.frames;
    }
    s_dispatch_since_render_ms = 0;

    GOBOT_PROFILE_FRAME("MainFrame");

    // SceneTree::Finalize can clear the simulation before Main::Cleanup runs.
    if (exit) WriteEditorBenchmark();
    return exit;

}

void Main::Cleanup() {
    WriteEditorBenchmark();
    OS::GetInstance()->DeleteMainLoop();

    Object::Delete(s_simulation_server);
    Object::Delete(s_physics_server);
    Object::Delete(s_input);
    Object::Delete(s_project_settings);

    python::PythonScriptRunner::Shutdown();
    SceneInitializer::Destroy();
}

}
