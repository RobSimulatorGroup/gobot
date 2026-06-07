#include "gobot/python/python_app_context.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/main/engine_context.hpp"
#include "gobot/physics/physics_server.hpp"
#include "gobot/rendering/headless_render_context.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/scene_initializer.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot::python {
namespace {

EngineContext* s_active_app_context = nullptr;
std::vector<EngineContext*>& AppContexts() {
    static auto* contexts = new std::vector<EngineContext*>();
    return *contexts;
}

void RegisterAppContext(EngineContext* context) {
    if (context == nullptr) {
        return;
    }
    auto& contexts = AppContexts();
    if (std::find(contexts.begin(), contexts.end(), context) == contexts.end()) {
        contexts.push_back(context);
    }
}

void UnregisterAppContext(EngineContext* context) {
    auto& contexts = AppContexts();
    contexts.erase(std::remove(contexts.begin(), contexts.end(), context), contexts.end());
}

const Node* RootForNode(const Node* node) {
    const Node* root = node;
    while (root != nullptr && root->GetParent() != nullptr) {
        root = root->GetParent();
    }
    return root;
}

struct OwnedAppContext {
    ProjectSettings* project_settings{nullptr};
    PhysicsServer* physics_server{nullptr};
    SimulationServer* simulation_server{nullptr};
    std::unique_ptr<EngineContext> context;

    OwnedAppContext() {
        project_settings = ProjectSettings::GetInstance();
        physics_server = Object::New<PhysicsServer>(PhysicsBackendType::Null, false);
        simulation_server = Object::New<SimulationServer>(PhysicsBackendType::Null, false);
        context = std::make_unique<EngineContext>(project_settings, physics_server, simulation_server);
        RegisterAppContext(context.get());
    }

    ~OwnedAppContext() {
        if (s_active_app_context == context.get()) {
            s_active_app_context = nullptr;
        }
        UnregisterAppContext(context.get());
        context.reset();
        if (simulation_server != nullptr) {
            Object::Delete(simulation_server);
            simulation_server = nullptr;
        }
        if (physics_server != nullptr) {
            Object::Delete(physics_server);
            physics_server = nullptr;
        }
    }
};

struct GobotRuntime {
    ProjectSettings* project_settings{nullptr};
    PhysicsServer* physics_server{nullptr};
    SimulationServer* app_simulation_server{nullptr};
    std::unique_ptr<EngineContext> app_context;
    std::unique_ptr<HeadlessRenderContext> headless_render_context;
    bool owns_project_settings{false};
    bool owns_physics_server{false};
    bool owns_simulation_server{false};
    bool scene_initializer_ready{false};

    GobotRuntime() {
        if (ProjectSettings::HasInstance()) {
            project_settings = ProjectSettings::GetInstance();
        } else {
            project_settings = Object::New<ProjectSettings>();
            owns_project_settings = true;
        }
        if (PhysicsServer::HasInstance()) {
            physics_server = PhysicsServer::GetInstance();
        } else {
            physics_server = Object::New<PhysicsServer>(PhysicsBackendType::Null, true);
            owns_physics_server = true;
        }
        if (SimulationServer::HasInstance()) {
            app_simulation_server = SimulationServer::GetInstance();
        } else {
            app_simulation_server = Object::New<SimulationServer>(PhysicsBackendType::Null, true);
            owns_simulation_server = true;
        }
        app_context = std::make_unique<EngineContext>(project_settings,
                                                      physics_server,
                                                      app_simulation_server);
        RegisterAppContext(app_context.get());
        SceneInitializer::Init();
        scene_initializer_ready = true;
    }

    ~GobotRuntime() {
        headless_render_context.reset();
        UnregisterAppContext(app_context.get());
        app_context.reset();
        if (owns_simulation_server && app_simulation_server != nullptr) {
            Object::Delete(app_simulation_server);
            app_simulation_server = nullptr;
        }
        if (scene_initializer_ready) {
            SceneInitializer::Destroy();
            scene_initializer_ready = false;
        }
        if (owns_physics_server && physics_server != nullptr) {
            Object::Delete(physics_server);
            physics_server = nullptr;
        }
        if (owns_project_settings && project_settings != nullptr) {
            Object::Delete(project_settings);
            project_settings = nullptr;
        }
    }

    EngineContext& GetAppContext() {
        return *app_context;
    }

    HeadlessRenderContext& EnsureHeadlessRenderContext() {
        if (!headless_render_context) {
            headless_render_context = std::make_unique<HeadlessRenderContext>();
        }
        return *headless_render_context;
    }

    void ShutdownHeadlessRenderContext() {
        headless_render_context.reset();
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

bool IsAppContextLive(const EngineContext* context) {
    const auto& contexts = AppContexts();
    return context != nullptr && std::find(contexts.begin(), contexts.end(), context) != contexts.end();
}

EngineContext* FindAppContextForSceneRoot(const Node* scene_node) {
    const Node* root = RootForNode(scene_node);
    if (root == nullptr) {
        return nullptr;
    }

    for (EngineContext* context : AppContexts()) {
        if (context != nullptr && RootForNode(context->GetSceneRoot()) == root) {
            return context;
        }
    }
    return nullptr;
}

std::shared_ptr<EngineContext> CreateAppContext() {
    Runtime();
    auto owner = std::make_shared<OwnedAppContext>();
    return std::shared_ptr<EngineContext>(owner, owner->context.get());
}

HeadlessRenderContext& EnsureHeadlessRenderContext() {
    return Runtime().EnsureHeadlessRenderContext();
}

void ShutdownHeadlessRenderContext() {
    Runtime().ShutdownHeadlessRenderContext();
}

} // namespace gobot::python
