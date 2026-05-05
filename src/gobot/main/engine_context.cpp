#include "gobot/main/engine_context.hpp"

#include <utility>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot {

EngineContext::EngineContext(ProjectSettings* project_settings,
                             PhysicsServer* physics_server,
                             SimulationServer* simulation_server)
    : project_settings_(project_settings),
      physics_server_(physics_server),
      simulation_server_(simulation_server) {
}

EngineContext::~EngineContext() {
    ClearWorld();
    ClearOwnedScene();
}

bool EngineContext::SetProjectPath(const std::string& project_path) {
    if (project_settings_ == nullptr) {
        SetLastError("ProjectSettings service is not available.");
        return false;
    }

    if (!project_settings_->SetProjectPath(project_path)) {
        SetLastError("Failed to set Gobot project path '" + project_path + "'.");
        return false;
    }

    last_error_.clear();
    return true;
}

const std::string& EngineContext::GetProjectPath() const {
    static const std::string empty;
    return project_settings_ == nullptr ? empty : project_settings_->GetProjectPath();
}

bool EngineContext::LoadScene(const std::string& scene_path) {
    if (load_scene_callback_) {
        if (!load_scene_callback_(scene_path)) {
            SetLastError("Failed to load scene through active runtime context from '" + scene_path + "'.");
            return false;
        }
        last_error_.clear();
        return true;
    }

    Ref<Resource> resource =
            ResourceLoader::Load(scene_path, "PackedScene", ResourceFormatLoader::CacheMode::Ignore);
    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource);
    if (!packed_scene.IsValid()) {
        SetLastError("Failed to load PackedScene from '" + scene_path + "'.");
        return false;
    }

    Node* scene_root = packed_scene->Instantiate();
    if (scene_root == nullptr) {
        SetLastError("Failed to instantiate PackedScene from '" + scene_path + "'.");
        return false;
    }

    SetSceneRoot(scene_root, true, scene_path);
    last_error_.clear();
    return true;
}

void EngineContext::SetSceneRoot(Node* scene_root, bool take_ownership, const std::string& scene_path) {
    if (scene_root_ == scene_root) {
        owns_scene_root_ = take_ownership;
        scene_path_ = scene_path;
        ClearWorld();
        return;
    }

    ClearWorld();
    ClearOwnedScene();
    scene_root_ = scene_root;
    owns_scene_root_ = take_ownership;
    scene_path_ = scene_path;
}

Node* EngineContext::GetSceneRoot() const {
    return scene_root_;
}

const std::string& EngineContext::GetScenePath() const {
    return scene_path_;
}

bool EngineContext::HasScene() const {
    return scene_root_ != nullptr;
}

void EngineContext::ClearScene() {
    ClearWorld();
    ClearOwnedScene();
    scene_root_ = nullptr;
    owns_scene_root_ = false;
    scene_path_.clear();
}

PhysicsBackendType EngineContext::GetBackendType() const {
    return simulation_server_ == nullptr ? PhysicsBackendType::Null : simulation_server_->GetBackendType();
}

void EngineContext::SetBackendType(PhysicsBackendType backend_type) {
    if (simulation_server_ != nullptr) {
        simulation_server_->SetBackendType(backend_type);
    }
}

bool EngineContext::BuildWorld() {
    if (simulation_server_ == nullptr) {
        SetLastError("SimulationServer service is not available.");
        return false;
    }
    if (scene_root_ == nullptr) {
        SetLastError("Cannot build physics world without a loaded scene.");
        return false;
    }

    if (!simulation_server_->BuildWorldFromScene(scene_root_)) {
        SetLastError(simulation_server_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool EngineContext::RebuildWorld(bool preserve_state) {
    if (simulation_server_ == nullptr) {
        SetLastError("SimulationServer service is not available.");
        return false;
    }
    if (scene_root_ == nullptr) {
        SetLastError("Cannot rebuild physics world without a loaded scene.");
        return false;
    }

    if (!simulation_server_->RebuildWorldFromScene(scene_root_, preserve_state)) {
        SetLastError(simulation_server_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

void EngineContext::ClearWorld() {
    if (simulation_server_ != nullptr) {
        simulation_server_->ClearWorld();
    }
}

bool EngineContext::HasWorld() const {
    return simulation_server_ != nullptr && simulation_server_->HasWorld();
}

bool EngineContext::ResetSimulation() {
    if (simulation_server_ == nullptr) {
        SetLastError("SimulationServer service is not available.");
        return false;
    }

    if (!simulation_server_->Reset()) {
        SetLastError(simulation_server_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool EngineContext::StepOnce() {
    if (simulation_server_ == nullptr) {
        SetLastError("SimulationServer service is not available.");
        return false;
    }

    if (!simulation_server_->StepOnce()) {
        SetLastError(simulation_server_->GetLastError());
        return false;
    }

    last_error_.clear();
    return true;
}

bool EngineContext::StepTicks(std::uint64_t ticks) {
    for (std::uint64_t tick = 0; tick < ticks; ++tick) {
        if (!StepOnce()) {
            return false;
        }
    }
    return true;
}

RealType EngineContext::GetSimulationTime() const {
    return simulation_server_ == nullptr ? RealType{0.0} : simulation_server_->GetSimulationTime();
}

std::uint64_t EngineContext::GetFrameCount() const {
    return simulation_server_ == nullptr ? 0 : simulation_server_->GetFrameCount();
}

SimulationServer* EngineContext::GetSimulationServer() const {
    return simulation_server_;
}

const std::string& EngineContext::GetLastError() const {
    return last_error_;
}

void EngineContext::SetSceneChangedCallback(SceneChangedCallback callback) {
    scene_changed_callback_ = std::move(callback);
}

void EngineContext::SetLoadSceneCallback(LoadSceneCallback callback) {
    load_scene_callback_ = std::move(callback);
}

void EngineContext::NotifySceneChanged() {
    if (scene_changed_callback_) {
        scene_changed_callback_();
    }
}

void EngineContext::ClearOwnedScene() {
    if (scene_root_ != nullptr && owns_scene_root_) {
        Object::Delete(scene_root_);
    }
    scene_root_ = nullptr;
    owns_scene_root_ = false;
}

void EngineContext::SetLastError(std::string error) {
    last_error_ = std::move(error);
}

} // namespace gobot
