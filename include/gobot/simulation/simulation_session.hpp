#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gobot {

// This contract deliberately depends only on the C++ standard library. A
// simulation executor must not borrow a SceneTree or an editor context.
struct SimulationMicrostepResult {
    bool completed{false};
    double advanced_time{0};
    bool state_valid{true};
    std::string error;
    std::string failure_stage;
    std::size_t failing_shard{std::numeric_limits<std::size_t>::max()};
    // False only for a healthy environment rolled back with a failed batch.
    // Such an environment can resume after the actual failing shard is reset.
    bool requires_reset{true};
};

struct SimulationEnvironmentProgress {
    std::uint64_t completed_microsteps{0};
    std::uint64_t completed_ticks{0};
    double advanced_time{0};
    bool completed{false};
    bool state_valid{true};
    std::string error;
    std::string failure_stage;
    std::size_t failing_shard{std::numeric_limits<std::size_t>::max()};
    bool requires_reset{false};
};

struct SimulationStepRequest {
    std::uint64_t ticks{1};
};

struct SimulationStepResult {
    std::vector<SimulationEnvironmentProgress> environments;

    bool Completed() const {
        return !environments.empty() && std::all_of(environments.begin(), environments.end(),
                [](const auto& result) { return result.completed && result.state_valid; });
    }
};

struct SimulationEnvironmentClock {
    std::uint64_t episode{0};
    std::uint64_t tick{0};
    std::uint64_t microstep{0};
    double time{0};
    bool faulted{false};
};

class SimulationExecutor {
public:
    virtual ~SimulationExecutor() = default;
    // Return one result per environment. A rolled-back microstep advances zero
    // time. Exceptions mean advancement is unknown and the state is invalid.
    virtual std::vector<SimulationMicrostepResult> Advance(double microstep_dt) = 0;
    virtual void Reset(const std::vector<std::size_t>& environments) = 0;
    virtual void Close() = 0;
};

class SimulationSession {
public:
    SimulationSession(std::shared_ptr<SimulationExecutor> executor,
                      std::size_t environment_count, double microstep_dt,
                      std::uint32_t substeps = 1)
        : executor_(std::move(executor)), clocks_(environment_count),
          microstep_dt_(microstep_dt), substeps_(substeps) {
        if (!executor_ || environment_count == 0 || !std::isfinite(microstep_dt_) ||
            microstep_dt_ <= 0 || substeps_ == 0 ||
            !std::isfinite(microstep_dt_ * substeps_)) {
            throw std::invalid_argument("Invalid simulation session configuration");
        }
    }

    SimulationSession(const SimulationSession&) = delete;
    SimulationSession& operator=(const SimulationSession&) = delete;
    ~SimulationSession() { try { Close(); } catch (...) {} }

    SimulationStepResult Step(SimulationStepRequest request = {}) {
        RequireOpen();
        if (request.ticks == 0 || request.ticks >
                std::numeric_limits<std::uint64_t>::max() / substeps_) {
            throw std::invalid_argument("Simulation tick count must be positive and representable");
        }
        if (std::any_of(clocks_.begin(), clocks_.end(), [](const auto& c) { return c.faulted; })) {
            throw std::logic_error("Reset faulted simulation environments before stepping");
        }
        SimulationStepResult result{std::vector<SimulationEnvironmentProgress>(clocks_.size())};
        for (std::uint64_t step = 0; step < request.ticks * substeps_; ++step) {
            std::vector<SimulationMicrostepResult> outputs;
            try {
                outputs = executor_->Advance(microstep_dt_);
                if (outputs.size() != clocks_.size()) {
                    throw std::runtime_error("Executor returned the wrong environment count");
                }
                // Validate the entire result before committing any clock.
                for (const auto& output : outputs) {
                    const double epsilon = std::max(1e-12, microstep_dt_ * 1e-5);
                    if (!std::isfinite(output.advanced_time) || output.advanced_time < 0 ||
                        output.advanced_time > microstep_dt_ + epsilon ||
                        (!output.completed && !output.requires_reset &&
                         (!output.state_valid || output.advanced_time != 0)) ||
                        (output.completed && (!output.state_valid || !output.error.empty() ||
                         std::abs(output.advanced_time - microstep_dt_) > epsilon))) {
                        throw std::runtime_error("Executor returned invalid simulation advancement");
                    }
                }
            } catch (const std::exception& error) {
                outputs.assign(clocks_.size(), {.state_valid = false, .error = error.what(),
                                                .failure_stage = "executor"});
            } catch (...) {
                outputs.assign(clocks_.size(), {.state_valid = false,
                        .error = "Unknown simulation executor failure", .failure_stage = "executor"});
            }
            bool failed = false;
            for (std::size_t env = 0; env < outputs.size(); ++env) {
                const auto& output = outputs[env];
                auto& progress = result.environments[env];
                auto& clock = clocks_[env];
                progress.advanced_time += output.advanced_time;
                clock.time += output.advanced_time;
                if (output.completed) {
                    ++progress.completed_microsteps;
                    ++clock.microstep;
                    if (clock.microstep % substeps_ == 0) {
                        ++progress.completed_ticks;
                        ++clock.tick;
                    }
                }
                if (!output.completed || !output.state_valid) {
                    failed = true;
                    clock.faulted = output.requires_reset;
                    progress.requires_reset = output.requires_reset;
                    progress.state_valid = output.state_valid;
                    progress.error = output.error.empty() ? "Simulation microstep did not complete" : output.error;
                    progress.failure_stage = output.failure_stage;
                    progress.failing_shard = output.failing_shard;
                }
            }
            if (failed) return result;
        }
        for (auto& progress : result.environments) progress.completed = true;
        return result;
    }

    void Reset(std::vector<std::size_t> environments = {}) {
        RequireOpen();
        if (environments.empty()) {
            for (std::size_t env = 0; env < clocks_.size(); ++env) environments.push_back(env);
        }
        auto sorted = environments;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.back() >= clocks_.size() ||
            std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("Reset environments must be unique valid indices");
        }
        try {
            executor_->Reset(environments);
        } catch (...) {
            for (auto env : environments) clocks_[env].faulted = true;
            throw;
        }
        for (auto env : environments) clocks_[env] = {.episode = clocks_[env].episode + 1};
    }

    void Close() {
        if (closed_) return;
        closed_ = true;
        executor_->Close();
    }

    const std::vector<SimulationEnvironmentClock>& GetClocks() const { return clocks_; }
    // Executor adapters call this only after successfully restoring the matching
    // backend checkpoint. It is not a public state-editing operation.
    void SetRestoredClocks(std::vector<SimulationEnvironmentClock> clocks) {
        RequireOpen();
        if (clocks.size() != clocks_.size() || std::any_of(clocks.begin(), clocks.end(),
                [](const auto& clock) { return !std::isfinite(clock.time) || clock.time < 0; })) {
            throw std::invalid_argument("Invalid restored simulation clocks");
        }
        clocks_ = std::move(clocks);
    }
    double GetFixedTimeStep() const { return microstep_dt_ * substeps_; }
    bool IsClosed() const { return closed_; }

private:
    void RequireOpen() const {
        if (closed_) throw std::logic_error("Simulation session is closed");
    }
    std::shared_ptr<SimulationExecutor> executor_;
    std::vector<SimulationEnvironmentClock> clocks_;
    double microstep_dt_;
    std::uint32_t substeps_;
    bool closed_{false};
};

} // namespace gobot
