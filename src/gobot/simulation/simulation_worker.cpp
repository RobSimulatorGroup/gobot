#include "gobot/simulation/simulation_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include "gobot/physics/physics_server.hpp"

namespace gobot {

struct SimulationWorker::Impl {
    mutable std::mutex mutex;
    std::condition_variable wake;
    bool shutdown = false;
    bool retiring = false;
    bool busy = false;
    bool installed = false;
    std::optional<Request> request;
    std::optional<Request> control;
    std::optional<Completion> completion;
    std::thread thread;

    Impl() : thread([this] { Run(); }) {}

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            shutdown = true;
        }
        wake.notify_one();
        thread.join();
    }

    static Completion Execute(Ref<PhysicsWorld>& world, Request& request) {
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
            if (request.operation == Operation::Step || request.operation == Operation::CaptureCheckpoint) {
                if (!ApplyPhysicsCommands(*world.Get(), request.commands, &frame->step.error)) {
                    frame->step.state_valid = false;
                    return result;
                }
            }
            if (request.operation == Operation::Step) {
                const RealType dt = request.settings.fixed_time_step;
                frame->step = world->Step(dt);
                const RealType tolerance = std::max(RealType(1e-8), dt * RealType(1e-5));
                if (!std::isfinite(frame->step.advanced_time) || frame->step.advanced_time < 0 ||
                    frame->step.advanced_time > dt + tolerance ||
                    (frame->step.completed && std::abs(frame->step.advanced_time - dt) > tolerance)) {
                    frame->step = {.state_valid = false, .error = "Physics backend returned an invalid step advancement."};
                }
                frame->simulation_time += frame->step.advanced_time;
                if (frame->step.completed && frame->step.state_valid) ++frame->tick;
            } else if (request.operation == Operation::Reset) {
                world->Reset();
                frame->step.completed = world->GetLastError().empty();
                frame->step.state_valid = frame->step.completed;
                frame->step.error = world->GetLastError();
            } else if (request.operation == Operation::RestoreCheckpoint) {
                frame->step.completed = world->RestoreCheckpoint(request.checkpoint.physics);
                frame->step.state_valid = frame->step.completed;
                frame->step.error = world->GetLastError();
                if (frame->step.completed) {
                    frame->tick = request.checkpoint.tick;
                    frame->simulation_time = request.checkpoint.simulation_time;
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

    void Run() {
        Ref<PhysicsWorld> world;
        std::unique_lock lock(mutex);
        for (;;) {
            wake.wait(lock, [&] { return shutdown || retiring || request || control; });
            if (shutdown || retiring) {
                const bool exit = shutdown;
                // Destruction can synchronize a device. Never do it on the UI thread or under the mailbox lock.
                auto abandoned = std::move(request);
                auto abandoned_control = std::move(control);
                request.reset();
                control.reset();
                completion.reset();
                busy = true;
                lock.unlock();
                abandoned.reset();
                abandoned_control.reset();
                world.Reset();
                lock.lock();
                installed = false;
                retiring = false;
                busy = false;
                if (exit) return;
                continue;
            }
            // Install must precede a reset submitted immediately after BuildWorld.
            const bool install_pending = request && request->operation == Operation::Install;
            Request job = control && !install_pending ? std::move(*control) : std::move(*request);
            if (control && !install_pending) {
                control.reset();
                request.reset();
            } else {
                request.reset();
            }
            busy = true;
            lock.unlock();
            auto output = Execute(world, job);
            job = {};
            lock.lock();
            installed = world.IsValid();
            busy = false;
            if (!retiring && !shutdown && !control) completion = std::move(output);
        }
    }
};

SimulationWorker::SimulationWorker() : impl_(std::make_unique<Impl>()) {}
SimulationWorker::~SimulationWorker() = default;

bool SimulationWorker::CanInstall() const {
    std::lock_guard lock(impl_->mutex);
    return !impl_->installed && !impl_->busy && !impl_->retiring && !impl_->request &&
            !impl_->control && !impl_->completion;
}

bool SimulationWorker::CanSubmit() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->installed && !impl_->busy && !impl_->retiring && !impl_->request &&
            !impl_->control && !impl_->completion;
}

bool SimulationWorker::IsPending() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->busy || impl_->retiring || impl_->request || impl_->control || impl_->completion;
}

bool SimulationWorker::Submit(Request request) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->busy || impl_->retiring || impl_->request || impl_->control || impl_->completion ||
        (request.operation == Operation::Install ? impl_->installed : !impl_->installed)) return false;
    impl_->request = std::move(request);
    impl_->wake.notify_one();
    return true;
}

bool SimulationWorker::RequestControl(Request request) {
    if (request.operation != Operation::Reset && request.operation != Operation::RestoreCheckpoint) return false;
    std::lock_guard lock(impl_->mutex);
    if (impl_->retiring || (!impl_->installed && !impl_->busy && !impl_->request) || impl_->control) return false;
    impl_->completion.reset();
    impl_->control = std::move(request);
    impl_->wake.notify_one();
    return true;
}

void SimulationWorker::Retire() {
    std::lock_guard lock(impl_->mutex);
    impl_->retiring = true;
    impl_->completion.reset();
    impl_->wake.notify_one();
}

std::optional<SimulationWorker::Completion> SimulationWorker::Poll() {
    std::lock_guard lock(impl_->mutex);
    auto result = std::move(impl_->completion);
    impl_->completion.reset();
    return result;
}

} // namespace gobot
