#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"

#include "gobot/core/config/engine.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/rl_environment.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace {

gobot::PhysicsBackendType ParseBackend(const std::string& backend) {
    if (backend == "null") {
        return gobot::PhysicsBackendType::Null;
    }
    if (backend == "mujoco") {
        return gobot::PhysicsBackendType::MuJoCoCpu;
    }
    throw std::invalid_argument("unknown physics backend '" + backend + "'");
}

} // namespace

int main(int argc, char* argv[]) {
    cxxopts::Options options("gobot_headless",
                             "Run Gobot scenes and RL environments without starting the editor UI.");
    options.add_options()
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

    if (!result.count("project")) {
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
                ParseBackend(result["backend"].as<std::string>()));
        gobot::SceneInitializer::Init();
        scene_initializer_ready = true;

        gobot::EngineContext context(project_settings, physics_server, simulation_server);
        if (!context.SetProjectPath(result["project"].as<std::string>())) {
            std::cerr << context.GetLastError() << std::endl;
            exit_code = 1;
        } else if (!context.LoadScene(result["scene"].as<std::string>())) {
            std::cerr << context.GetLastError() << std::endl;
            exit_code = 1;
        } else {
            std::cout << "loaded scene root=" << context.GetSceneRoot()->GetName()
                      << " type=" << context.GetSceneRoot()->GetClassStringName()
                      << std::endl;

            const std::string robot = result["robot"].as<std::string>();
            if (!robot.empty()) {
                gobot::RLEnvironment environment(simulation_server);
                environment.SetSceneRoot(context.GetSceneRoot());
                environment.SetRobotName(robot);
                gobot::RLEnvironmentResetResult reset_result =
                        environment.Reset(result["seed"].as<std::uint32_t>());
                if (!reset_result.ok) {
                    std::cerr << reset_result.error << std::endl;
                    exit_code = 1;
                } else {
                    std::cout << "reset ok=true observation_size="
                              << reset_result.observation.size()
                              << " action_size=" << environment.GetActionSize()
                              << std::endl;
                    std::vector<gobot::RealType> action(environment.GetActionSize(), 0.0);
                    for (std::uint64_t step = 0; step < result["steps"].as<std::uint64_t>(); ++step) {
                        gobot::RLEnvironmentStepResult step_result = environment.Step(action);
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
            } else if (!context.BuildWorld()) {
                std::cerr << context.GetLastError() << std::endl;
                exit_code = 1;
            } else if (!context.ResetSimulation()) {
                std::cerr << context.GetLastError() << std::endl;
                exit_code = 1;
            } else if (!context.StepTicks(result["steps"].as<std::uint64_t>())) {
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
