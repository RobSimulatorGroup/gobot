#pragma once

#include "gobot/physics/physics_types.hpp"

namespace gobot {

// Immutable after publication. Retaining a frame never retains a live physics world.
struct SimulationStateFrame {
    std::uint64_t epoch = 0;
    std::uint64_t tick = 0;
    RealType simulation_time = 0;
    PhysicsStepResult step;
    PhysicsSceneState state;
};

} // namespace gobot
