#pragma once

#include "gobot/physics/physics_world.hpp"
#include "gobot/simulation/simulation_session.hpp"

namespace gobot {

// Adapter for the native, single-scene execution path. The standalone session
// contract has no dependency on this adapter or on PhysicsWorld.
class PhysicsWorldExecutor final : public SimulationExecutor {
public:
    explicit PhysicsWorldExecutor(Ref<PhysicsWorld> world) : world_(std::move(world)) {}
    std::vector<SimulationMicrostepResult> Advance(double dt) override {
        last_step = {};
        last_step = world_->Step(static_cast<RealType>(dt));
        return {{last_step.completed, last_step.advanced_time, last_step.state_valid, last_step.error}};
    }
    void Reset(const std::vector<std::size_t>& environments) override {
        if (environments != std::vector<std::size_t>{0}) throw std::invalid_argument("Invalid native scene environment");
        world_->Reset();
        if (!world_->GetLastError().empty()) throw std::runtime_error(world_->GetLastError());
    }
    void Close() override { world_.Reset(); }
    const Ref<PhysicsWorld>& GetWorld() const { return world_; }
    PhysicsStepResult last_step;

    PhysicsStepResult Resolve(const SimulationStepResult& result) const {
        const auto& progress = result.environments.front();
        PhysicsStepResult step = last_step;
        step.completed = progress.completed;
        step.advanced_time = progress.advanced_time;
        step.state_valid = progress.state_valid;
        step.error = progress.error;
        return step;
    }
private:
    Ref<PhysicsWorld> world_;
};

} // namespace gobot
