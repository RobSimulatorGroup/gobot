#pragma once

#include <cstdint>
#include <string>

#include "gobot/core/object.hpp"
#include "gobot/physics/physics_server.hpp"

namespace gobot {

class Node;
class PhysicsServer;
class ProjectSettings;
class SimulationServer;

class GOBOT_EXPORT EngineContext {
public:
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

    SimulationServer* GetSimulationServer() const;
    const std::string& GetLastError() const;

private:
    void ClearOwnedScene();
    void SetLastError(std::string error);

    ProjectSettings* project_settings_{nullptr};
    PhysicsServer* physics_server_{nullptr};
    SimulationServer* simulation_server_{nullptr};
    Node* scene_root_{nullptr};
    bool owns_scene_root_{false};
    std::string scene_path_;
    std::string last_error_;
};

} // namespace gobot
