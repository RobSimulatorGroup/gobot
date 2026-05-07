#include "gobot/python/python_app_context.hpp"

#include <memory>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot::python {
namespace {

EngineContext* s_active_app_context = nullptr;

struct GobotRuntime {
    ProjectSettings* project_settings{nullptr};
    PhysicsServer* physics_server{nullptr};
    SimulationServer* app_simulation_server{nullptr};
    std::unique_ptr<EngineContext> app_context;
    bool scene_initializer_ready{false};

    GobotRuntime() {
        project_settings = Object::New<ProjectSettings>();
        physics_server = Object::New<PhysicsServer>();
        app_simulation_server = Object::New<SimulationServer>();
        app_context = std::make_unique<EngineContext>(project_settings,
                                                      physics_server,
                                                      app_simulation_server);
        SceneInitializer::Init();
        scene_initializer_ready = true;
    }

    ~GobotRuntime() {
        app_context.reset();
        if (app_simulation_server != nullptr) {
            Object::Delete(app_simulation_server);
            app_simulation_server = nullptr;
        }
        if (scene_initializer_ready) {
            SceneInitializer::Destroy();
            scene_initializer_ready = false;
        }
        if (physics_server != nullptr) {
            Object::Delete(physics_server);
            physics_server = nullptr;
        }
        if (project_settings != nullptr) {
            Object::Delete(project_settings);
            project_settings = nullptr;
        }
    }

    EngineContext& GetAppContext() {
        return *app_context;
    }
};

GobotRuntime& Runtime() {
    static GobotRuntime runtime;
    return runtime;
}

} // namespace

void SetActiveAppContext(EngineContext* context) {
    s_active_app_context = context;
}

EngineContext& GetActiveAppContext() {
    if (s_active_app_context != nullptr) {
        return *s_active_app_context;
    }
    return Runtime().GetAppContext();
}

EngineContext* GetActiveAppContextOrNull() {
    return s_active_app_context;
}

} // namespace gobot::python
