#pragma once

#include <memory>
#include <optional>
#include "gobot/physics/physics_commands.hpp"
#include "gobot/physics/physics_world.hpp"
#include "gobot/simulation/simulation_state_frame.hpp"

namespace gobot {

struct SimulationCheckpoint {
    Ref<PhysicsRuntimeCheckpoint> physics;
    std::uint64_t tick = 0;
    RealType simulation_time = 0;
};

// One worker, one request and one completed frame. Only shutdown joins the thread.
class GOBOT_EXPORT SimulationWorker {
public:
    enum class Operation { Install, Step, Reset, CaptureCheckpoint, RestoreCheckpoint };
    struct Request {
        Operation operation = Operation::Step;
        std::uint64_t epoch = 0;
        std::uint64_t tick = 0;
        RealType simulation_time = 0;
        PhysicsWorldSettings settings;
        PhysicsCommands commands;
        PhysicsBackendType backend = PhysicsBackendType::Null;
        PhysicsSceneSnapshot install_snapshot;
        SimulationCheckpoint checkpoint;
    };
    struct Completion {
        Operation operation;
        std::shared_ptr<const SimulationStateFrame> frame;
        SimulationCheckpoint checkpoint;
        PhysicsBackendCapabilities capabilities;
        std::optional<PhysicsSceneArtifact> artifact;
    };

    SimulationWorker();
    ~SimulationWorker();
    SimulationWorker(const SimulationWorker&) = delete;
    SimulationWorker& operator=(const SimulationWorker&) = delete;

    bool CanInstall() const;
    bool CanSubmit() const;
    bool IsPending() const;
    bool Submit(Request request);
    // Reset/restore replace queued work at the next tick boundary, including while a solve is running.
    bool RequestControl(Request request);
    void Retire();
    std::optional<Completion> Poll();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gobot
