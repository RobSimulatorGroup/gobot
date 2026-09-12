#include "gobot/simulation/simulation_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include "gobot/physics/physics_server.hpp"
#include "gobot/simulation/simulation_session.hpp"
#include "gobot/simulation/physics_world_executor.hpp"
#include "gobot/simulation/simulation_task_worker.hpp"

namespace gobot {

namespace {
struct NativeSimulationRuntime {
    using Request = SimulationWorker::Request;
    using Completion = SimulationWorker::Completion;
    using Operation = SimulationWorker::Operation;
    Ref<PhysicsWorld> world;
    std::shared_ptr<PhysicsWorldExecutor> executor;
    std::unique_ptr<SimulationSession> session;

    static bool IsInstall(const Request& request) { return request.operation == Operation::Install; }
    static bool IsControl(const Request& request) {
        return request.operation == Operation::Reset || request.operation == Operation::RestoreCheckpoint;
    }
    bool IsInstalled() const { return world.IsValid(); }
    void Retire() noexcept {
        session.reset();
        executor.reset();
        world.Reset();
    }

    Completion Execute(Request& request) {
        auto frame = std::make_shared<SimulationStateFrame>();
        frame->epoch = request.epoch;
        frame->tick = request.tick;
        frame->simulation_time = request.simulation_time;
        Completion result{request.operation, frame, {}, {}, {}};
        try {
            const auto started = std::chrono::steady_clock::now();
            if (request.operation == Operation::Install) {
                // SDK construction and destruction must use the same thread as the solver.
                world = PhysicsServer::CreateWorld(request.backend, request.settings);
                if (world.IsValid() && (world->GetBackendType() != request.backend || !world->IsAvailable())) {
                    world.Reset();
                }
                if (world.IsValid() && !world->Build(std::move(request.install_snapshot))) {
                    frame->step.state_valid = false;
                    frame->step.error = world->GetLastError();
                    if (frame->step.error.empty()) frame->step.error = "Physics backend could not build the scene.";
                    world.Reset();
                    return result;
                }
            }
            if (!world.IsValid()) {
                frame->step.state_valid = false;
                frame->step.error = "Simulation worker has no installed physics world.";
                return result;
            }
            world->SetSettings(request.settings);
            if (!session || session->GetFixedTimeStep() != request.settings.fixed_time_step) {
                const std::vector<SimulationEnvironmentClock> clocks{{
                        .tick = request.tick, .microstep = request.tick,
                        .time = request.simulation_time}};
                session.reset();
                executor = std::make_shared<PhysicsWorldExecutor>(world);
                session = std::make_unique<SimulationSession>(executor, 1, request.settings.fixed_time_step);
                session->SetRestoredClocks(clocks);
            }
            if (request.operation == Operation::Step || request.operation == Operation::CaptureCheckpoint) {
                if (!ApplyPhysicsCommands(*world.Get(), request.commands, &frame->step.error)) {
                    frame->step.state_valid = false;
                    return result;
                }
            }
            if (request.operation == Operation::Step) {
                const auto result = session->Step();
                frame->step = executor->Resolve(result);
                frame->simulation_time = session->GetClocks().front().time;
                frame->tick = session->GetClocks().front().tick;
            } else if (request.operation == Operation::Reset) {
                session->Reset();
                frame->step.completed = true;
                frame->simulation_time = 0;
                frame->tick = 0;
            } else if (request.operation == Operation::RestoreCheckpoint) {
                frame->step.completed = world->RestoreCheckpoint(request.checkpoint.physics);
                frame->step.state_valid = frame->step.completed;
                frame->step.error = world->GetLastError();
                if (frame->step.completed) {
                    frame->tick = request.checkpoint.tick;
                    frame->simulation_time = request.checkpoint.simulation_time;
                    session->SetRestoredClocks({{.tick = frame->tick, .microstep = frame->tick,
                                                .time = frame->simulation_time}});
                }
            } else if (request.operation == Operation::CaptureCheckpoint) {
                result.checkpoint = {world->CaptureCheckpoint(), request.tick, request.simulation_time};
                frame->step.completed = result.checkpoint.physics.IsValid();
                if (!frame->step.completed) frame->step.error = "Physics backend could not capture a runtime checkpoint.";
            } else {
                frame->step.completed = true;
            }
            if (frame->step.completed && (request.operation == Operation::Install ||
                request.operation == Operation::Reset || request.operation == Operation::RestoreCheckpoint)) {
                result.capabilities = world->GetCapabilities();
                if (const auto* artifact = world->GetSceneArtifact()) result.artifact = *artifact;
            }
            frame->step.diagnostics = world->GetSolverDiagnostics();
            frame->step.diagnostics.total_step_time_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (frame->step.state_valid) frame->state = world->GetSceneState();
        } catch (const std::exception& error) {
            frame->step.completed = false;
            frame->step.state_valid = false;
            frame->step.error = error.what();
        } catch (...) {
            frame->step.completed = false;
            frame->step.state_valid = false;
            frame->step.error = "Simulation worker caught an unknown backend exception.";
        }
        if (request.operation == Operation::Install && !frame->step.completed) world.Reset();
        return result;
    }

};
} // namespace

struct SimulationWorker::Impl : SimulationTaskWorker<NativeSimulationRuntime> {};

SimulationWorker::SimulationWorker() : impl_(std::make_unique<Impl>()) {}
SimulationWorker::~SimulationWorker() = default;

bool SimulationWorker::CanInstall() const {
    return impl_->CanInstall();
}

bool SimulationWorker::CanSubmit() const {
    return impl_->CanSubmit();
}

bool SimulationWorker::IsPending() const {
    return impl_->IsPending();
}

bool SimulationWorker::Submit(Request request) {
    return impl_->Submit(std::move(request));
}

bool SimulationWorker::RequestControl(Request request) {
    return impl_->RequestControl(std::move(request));
}

void SimulationWorker::Retire() {
    impl_->Retire();
}

std::optional<SimulationWorker::Completion> SimulationWorker::Poll() {
    return impl_->Poll();
}

} // namespace gobot
