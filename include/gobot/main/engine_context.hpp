#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_types.hpp"
#include "gobot/physics/physics_server.hpp"

namespace gobot {

class Node;
class PhysicsServer;
class ProjectSettings;
class SimulationServer;

class GOBOT_EXPORT EngineContext {
public:
    using SceneChangedCallback = std::function<void()>;
    using LoadSceneCallback = std::function<bool(const std::string& path)>;

    EngineContext(ProjectSettings* project_settings,
                  PhysicsServer* physics_server,
                  SimulationServer* simulation_server);

    ~EngineContext();

    EngineContext(const EngineContext&) = delete;
    EngineContext& operator=(const EngineContext&) = delete;

    bool SetProjectPath(const std::string& project_path);
    const std::string& GetProjectPath() const;

    bool LoadScene(const std::string& scene_path);
    void SetSceneRoot(Node* scene_root,
                      bool take_ownership = false,
                      const std::string& scene_path = "");
    Node* GetSceneRoot() const;
    const std::string& GetScenePath() const;
    std::uint64_t GetSceneEpoch() const;
    bool HasScene() const;
    void ClearScene();

    PhysicsBackendType GetBackendType() const;
    void SetBackendType(PhysicsBackendType backend_type);

    bool BuildWorld();
    bool RebuildWorld(bool preserve_state = true);
    void ClearWorld();
    bool HasWorld() const;

    bool ResetSimulation();
    bool StepOnce();
    bool StepTicks(std::uint64_t ticks);

    RealType GetSimulationTime() const;
    std::uint64_t GetFrameCount() const;
    Vector3 GetGravity() const;

    SimulationServer* GetSimulationServer() const;
    const std::string& GetLastError() const;

    void SetSceneChangedCallback(SceneChangedCallback callback);
    void SetLoadSceneCallback(LoadSceneCallback callback);
    void NotifySceneChanged();
    void NotifySceneMutated();

private:
    void ClearOwnedScene();
    void AdvanceSceneEpoch();
    void SetLastError(std::string error);

    ProjectSettings* project_settings_{nullptr};
    PhysicsServer* physics_server_{nullptr};
    SimulationServer* simulation_server_{nullptr};
    Node* scene_root_{nullptr};
    bool owns_scene_root_{false};
    // Monotonically changes when the active scene root is replaced or cleared.
    // Python node handles capture this value so handles from an old scene fail
    // with ReferenceError instead of accidentally resolving into a new scene.
    std::uint64_t scene_epoch_{1};
    std::string scene_path_;
    std::string last_error_;
    SceneChangedCallback scene_changed_callback_;
    LoadSceneCallback load_scene_callback_;
};

} // namespace gobot
