#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include <nlohmann/json.hpp>

#include "gobot/core/config/engine.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/python/python_binding_registry.hpp"
#include "gobot/python/python_script_runner.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/rl_environment.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace {

struct HeadlessOptions {
    std::string project;
    std::string scene{"res://world.jscn"};
    std::string backend{"null"};
    std::string robot;
    std::string script;
    std::uint64_t steps{1};
    std::uint32_t seed{1};
};

gobot::PhysicsBackendType ParseBackend(const std::string& backend) {
    if (backend == "null") {
        return gobot::PhysicsBackendType::Null;
    }
    if (backend == "mujoco") {
        return gobot::PhysicsBackendType::MuJoCoCpu;
    }
    throw std::invalid_argument("unknown physics backend '" + backend + "'");
}

bool LoadConfig(const std::string& path, HeadlessOptions* options, std::string* error) {
    std::ifstream stream(path);
    if (!stream) {
        *error = "Cannot open headless config: " + path;
        return false;
    }

    nlohmann::json config;
    try {
        stream >> config;
    } catch (const std::exception& exception) {
        *error = std::string("Cannot parse headless config: ") + exception.what();
        return false;
    }

    if (config.contains("project")) {
        options->project = config["project"].get<std::string>();
    }
    if (config.contains("scene")) {
        options->scene = config["scene"].get<std::string>();
    }
    if (config.contains("backend")) {
        options->backend = config["backend"].get<std::string>();
    }
    if (config.contains("robot")) {
        options->robot = config["robot"].get<std::string>();
    }
    if (config.contains("script")) {
        options->script = config["script"].get<std::string>();
    }
    if (config.contains("steps")) {
        options->steps = config["steps"].get<std::uint64_t>();
    }
    if (config.contains("seed")) {
        options->seed = config["seed"].get<std::uint32_t>();
    }

    return true;
}

void ApplyCliOptions(const cxxopts::ParseResult& result, HeadlessOptions* options) {
    if (result.count("project")) {
        options->project = result["project"].as<std::string>();
    }
    if (result.count("scene")) {
        options->scene = result["scene"].as<std::string>();
    }
    if (result.count("backend")) {
        options->backend = result["backend"].as<std::string>();
    }
    if (result.count("robot")) {
        options->robot = result["robot"].as<std::string>();
    }
    if (result.count("script")) {
        options->script = result["script"].as<std::string>();
    }
    if (result.count("steps")) {
        options->steps = result["steps"].as<std::uint64_t>();
    }
    if (result.count("seed")) {
        options->seed = result["seed"].as<std::uint32_t>();
    }
}

} // namespace

int main(int argc, char* argv[]) {
    cxxopts::Options options("gobot_headless",
                             "Run Gobot scenes and RL environments without starting the editor UI.");
    options.add_options()
            ("config", "JSON headless training/run config", cxxopts::value<std::string>())
            ("project", "Gobot project path", cxxopts::value<std::string>())
            ("scene", "Scene path, usually res://world.jscn",
             cxxopts::value<std::string>()->default_value("res://world.jscn"))
            ("backend", "Physics backend: null or mujoco",
             cxxopts::value<std::string>()->default_value("null"))
            ("robot", "Robot name used for RL smoke stepping",
             cxxopts::value<std::string>()->default_value(""))
            ("steps", "Number of fixed simulation ticks to run",
             cxxopts::value<std::uint64_t>()->default_value("1"))
            ("seed", "RL reset seed",
             cxxopts::value<std::uint32_t>()->default_value("1"))
            ("script", "Python script to execute inside the headless Gobot runtime",
             cxxopts::value<std::string>())
            ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return 2;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    HeadlessOptions headless_options;
    if (result.count("config")) {
        std::string error;
        if (!LoadConfig(result["config"].as<std::string>(), &headless_options, &error)) {
            std::cerr << error << std::endl;
            return 2;
        }
    }
    ApplyCliOptions(result, &headless_options);

    if (headless_options.project.empty()) {
        std::cerr << "--project is required" << std::endl;
        std::cerr << options.help() << std::endl;
        return 2;
    }

    gobot::Engine* engine = gobot::Object::New<gobot::Engine>();
    gobot::ProjectSettings* project_settings = gobot::Object::New<gobot::ProjectSettings>();
    gobot::PhysicsServer* physics_server = gobot::Object::New<gobot::PhysicsServer>();
    gobot::SimulationServer* simulation_server = nullptr;

    bool scene_initializer_ready = false;
    int exit_code = 0;

    try {
        simulation_server = gobot::Object::New<gobot::SimulationServer>(
                ParseBackend(headless_options.backend));
        gobot::SceneInitializer::Init();
        scene_initializer_ready = true;

        gobot::EngineContext context(project_settings, physics_server, simulation_server);
        gobot::python::SetActiveAppContext(&context);

        if (!context.SetProjectPath(headless_options.project)) {
            std::cerr << context.GetLastError() << std::endl;
            exit_code = 1;
        } else if (!context.LoadScene(headless_options.scene)) {
            std::cerr << context.GetLastError() << std::endl;
            exit_code = 1;
        } else {
            std::cout << "loaded scene root=" << context.GetSceneRoot()->GetName()
                      << " type=" << context.GetSceneRoot()->GetClassStringName()
                      << std::endl;

            if (!headless_options.script.empty()) {
                gobot::python::PythonExecutionResult script_result =
                        gobot::python::PythonScriptRunner::ExecuteFile(headless_options.script, &context);
                if (!script_result.output.empty()) {
                    std::cout << script_result.output;
                    if (script_result.output.back() != '\n') {
                        std::cout << std::endl;
                    }
                }
                if (!script_result.error.empty()) {
                    std::cerr << script_result.error;
                    if (script_result.error.back() != '\n') {
                        std::cerr << std::endl;
                    }
                }
                exit_code = script_result.ok ? 0 : 1;
            } else if (!headless_options.robot.empty()) {
                gobot::RLEnvironment* environment =
                        gobot::Object::New<gobot::RLEnvironment>(simulation_server);
                environment->SetSceneRoot(context.GetSceneRoot());
                environment->SetRobotName(headless_options.robot);
                gobot::RLEnvironmentResetResult reset_result =
                        environment->Reset(headless_options.seed);
                if (!reset_result.ok) {
                    std::cerr << reset_result.error << std::endl;
                    exit_code = 1;
                } else {
                    std::cout << "reset ok=true observation_size="
                              << reset_result.observation.size()
                              << " action_size=" << environment->GetActionSize()
                              << std::endl;
                    std::vector<gobot::RealType> action(environment->GetActionSize(), 0.0);
                    for (std::uint64_t step = 0; step < headless_options.steps; ++step) {
                        gobot::RLEnvironmentStepResult step_result = environment->Step(action);
                        if (!step_result.error.empty()) {
                            std::cerr << step_result.error << std::endl;
                            exit_code = 1;
                            break;
                        }
                        std::cout << "step=" << (step + 1)
                                  << " reward=" << step_result.reward
                                  << " terminated=" << (step_result.terminated ? "true" : "false")
                                  << " truncated=" << (step_result.truncated ? "true" : "false")
                                  << " frame=" << step_result.frame_count
                                  << std::endl;
                        if (step_result.terminated || step_result.truncated) {
                            break;
                        }
                    }
                }
                gobot::Object::Delete(environment);
            } else if (!context.BuildWorld()) {
                std::cerr << context.GetLastError() << std::endl;
                exit_code = 1;
            } else if (!context.ResetSimulation()) {
                std::cerr << context.GetLastError() << std::endl;
                exit_code = 1;
            } else if (!context.StepTicks(headless_options.steps)) {
                std::cerr << context.GetLastError() << std::endl;
                exit_code = 1;
            } else {
                std::cout << "simulation frame=" << context.GetFrameCount()
                          << " time=" << context.GetSimulationTime()
                          << std::endl;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        exit_code = 1;
    }

    if (gobot::python::GetActiveAppContextOrNull() != nullptr) {
        gobot::python::SetActiveAppContext(nullptr);
    }
    gobot::python::PythonScriptRunner::Shutdown();
    if (scene_initializer_ready) {
        gobot::SceneInitializer::Destroy();
    }
    if (simulation_server != nullptr) {
        gobot::Object::Delete(simulation_server);
    }
    gobot::Object::Delete(physics_server);
    gobot::Object::Delete(project_settings);
    gobot::Object::Delete(engine);
    return exit_code;
}
