#pragma once

#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <limits>
#include <set>
#include <string>
#include "gobot/simulation/simulation_session.hpp"
#include "gobot/simulation/simulation_task_worker.hpp"

namespace gobot {

// CPU presentation/command data only. Device arrays stay inside the executor;
// these selected fields cross the runtime boundary at publication cadence.
struct SimulationBuffer {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<double> values;
};
struct SimulationSubscription {
    std::vector<std::string> fields;
    std::vector<std::size_t> environments{0};
    double interval_seconds{1.0 / 60.0};
};
struct SimulationSnapshot {
    std::uint64_t epoch{0};
    std::vector<std::size_t> environments;
    std::vector<SimulationEnvironmentClock> clocks;
    std::vector<SimulationBuffer> buffers;
};
struct SimulationRuntimeInfo {
    std::string provider_name;
    std::string device;
    std::string graph_status{"Disabled"};
    std::size_t environment_count{0};
    double fixed_time_step{0};
    std::size_t controlled_joint_count{0};
    std::map<std::string, std::uint64_t> capacities;
};
class SimulationDataExecutor : public SimulationExecutor {
public:
    virtual void ApplyCommands(const std::vector<SimulationBuffer>& commands) = 0;
    virtual void CaptureSnapshot(const SimulationSubscription& subscription,
                                 std::vector<SimulationBuffer>& buffers) = 0;
};
struct SimulationExecutorInstallation {
    std::shared_ptr<SimulationDataExecutor> executor;
    std::size_t environment_count{1};
    double microstep_dt{.002};
    std::uint32_t substeps{1};
    SimulationRuntimeInfo info;
};

// Factory captures immutable configuration/artifacts only. Its executor is
// constructed and destroyed by SimulationTaskWorker, never by the editor.
struct SimulationDataRuntime {
    enum class Operation { Install, Step, Reset, Subscribe };
    struct Request {
        Operation operation{Operation::Step};
        std::uint64_t epoch{0};
        std::uint64_t ticks{1};
        std::function<SimulationExecutorInstallation()> factory;
        std::vector<SimulationBuffer> commands;
        std::vector<std::size_t> reset_environments;
        SimulationSubscription subscription;
    };
    struct Completion {
        Operation operation;
        std::uint64_t epoch{0};
        SimulationStepResult step;
        std::vector<SimulationEnvironmentClock> clocks;
        std::shared_ptr<const SimulationSnapshot> snapshot;
        std::string error;
        std::string presentation_error;
        double elapsed_seconds{0};
        bool snapshot_skipped{false};
        SimulationRuntimeInfo info;
    };

    static bool IsInstall(const Request& request) { return request.operation == Operation::Install; }
    static bool IsControl(const Request& request) { return request.operation == Operation::Reset; }
    bool IsInstalled() const { return bool(session_); }
    void Retire() noexcept {
        // Session catches close failures during destruction. Releasing the
        // final executor reference here also releases all SDK/device storage.
        if (!session_ && executor_) {
            try { executor_->Close(); } catch (...) {}
        }
        session_.reset();
        executor_.reset();
        for (auto& slot : snapshots_) slot.reset();
    }

    Completion Execute(Request& request) {
        const auto started = std::chrono::steady_clock::now();
        Completion result{request.operation, request.epoch};
        bool publish = false;
        try {
            if (IsInstall(request)) {
                auto installation = request.factory();
                executor_ = std::move(installation.executor);
                session_ = std::make_unique<SimulationSession>(executor_, installation.environment_count,
                        installation.microstep_dt, installation.substeps);
                info_ = std::move(installation.info);
                info_.environment_count = installation.environment_count;
                info_.fixed_time_step = session_->GetFixedTimeStep();
                ValidateSubscription(request.subscription);
                subscription_ = request.subscription;
                publish = true;
            } else if (!session_) {
                throw std::logic_error("Simulation worker has no installed session");
            } else if (request.operation == Operation::Reset) {
                session_->Reset(request.reset_environments);
                publish = true;
            } else if (request.operation == Operation::Subscribe) {
                ValidateSubscription(request.subscription);
                subscription_ = request.subscription;
                publish = true;
            } else {
                executor_->ApplyCommands(request.commands);
                result.step = session_->Step({request.ticks});
                const bool valid = std::all_of(result.step.environments.begin(), result.step.environments.end(),
                        [](const auto& env) { return env.state_valid; });
                publish = valid && std::chrono::duration<double>(started - last_publication_).count() >=
                        subscription_.interval_seconds;
            }
        } catch (const std::exception& error) {
            result.error = error.what();
            if (IsInstall(request)) Retire();
            else if (request.operation == Operation::Step) FaultSession();
        } catch (...) {
            result.error = "Unknown simulation runtime failure";
            if (IsInstall(request)) Retire();
            else if (request.operation == Operation::Step) FaultSession();
        }
        if (session_) {
            result.clocks = session_->GetClocks();
            result.info = info_;
        }
        if (publish && result.error.empty()) {
            // A retained buffer view keeps its whole frame leased. Skip only
            // presentation when all three slots are held; simulation continues.
            auto slot = std::find_if(snapshots_.begin(), snapshots_.end(),
                    [](const auto& frame) { return !frame || frame.use_count() == 1; });
            if (slot == snapshots_.end()) {
                result.snapshot_skipped = true;
            } else {
                if (!*slot) *slot = std::make_shared<SimulationSnapshot>();
                auto& frame = **slot;
                try {
                    frame.buffers.clear();
                    executor_->CaptureSnapshot(subscription_, frame.buffers);
                    ValidateSnapshot(frame.buffers);
                    frame.epoch = request.epoch;
                    frame.environments = subscription_.environments;
                    frame.clocks.clear();
                    for (auto env : subscription_.environments) frame.clocks.push_back(result.clocks.at(env));
                    result.snapshot = *slot;
                    last_publication_ = std::chrono::steady_clock::now();
                } catch (const std::exception& error) {
                    result.presentation_error = error.what();
                } catch (...) {
                    result.presentation_error = "Unknown simulation snapshot failure";
                }
            }
        }
        result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return result;
    }

private:
    void ValidateSnapshot(const std::vector<SimulationBuffer>& buffers) const {
        std::set<std::string> remaining(subscription_.fields.begin(), subscription_.fields.end());
        for (const auto& buffer : buffers) {
            if (!remaining.erase(buffer.name) || buffer.shape.empty() ||
                buffer.shape.front() != subscription_.environments.size())
                throw std::invalid_argument("Snapshot fields or environment dimensions differ from subscription");
            std::size_t bytes = sizeof(double);
            for (auto it = buffer.shape.rbegin(); it != buffer.shape.rend(); ++it) {
                const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
                if (*it > limit || (bytes && *it > limit / bytes))
                    throw std::invalid_argument("Snapshot dimensions overflow buffer strides");
                bytes *= *it;
            }
            if (bytes / sizeof(double) != buffer.values.size() ||
                !std::all_of(buffer.values.begin(), buffer.values.end(), [](double v) { return std::isfinite(v); }))
                throw std::invalid_argument("Snapshot storage does not match its dimensions or contains non-finite values");
        }
        if (!remaining.empty()) throw std::invalid_argument("Snapshot is missing subscribed fields");
    }

    void FaultSession() {
        if (!session_) return;
        auto clocks = session_->GetClocks();
        for (auto& clock : clocks) clock.faulted = true;
        session_->SetRestoredClocks(std::move(clocks));
    }
    void ValidateSubscription(const SimulationSubscription& subscription) const {
        if (!std::isfinite(subscription.interval_seconds) || subscription.interval_seconds < 0)
            throw std::invalid_argument("Snapshot interval must be finite and non-negative");
        auto indices = subscription.environments;
        std::sort(indices.begin(), indices.end());
        if ((!indices.empty() && indices.back() >= session_->GetClocks().size()) ||
            std::adjacent_find(indices.begin(), indices.end()) != indices.end())
            throw std::invalid_argument("Snapshot environments must be unique valid indices");
        auto fields = subscription.fields;
        std::sort(fields.begin(), fields.end());
        if (std::any_of(fields.begin(), fields.end(), [](const auto& field) { return field.empty(); }) ||
            std::adjacent_find(fields.begin(), fields.end()) != fields.end())
            throw std::invalid_argument("Snapshot fields must be unique non-empty names");
    }

    std::shared_ptr<SimulationDataExecutor> executor_;
    std::unique_ptr<SimulationSession> session_;
    SimulationSubscription subscription_;
    SimulationRuntimeInfo info_;
    std::array<std::shared_ptr<SimulationSnapshot>, 3> snapshots_;
    std::chrono::steady_clock::time_point last_publication_{};
};

using SimulationDataWorker = SimulationTaskWorker<SimulationDataRuntime>;
} // namespace gobot
