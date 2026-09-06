#include <gtest/gtest.h>

#include <limits>

#include "gobot/physics/backends/null_physics_world.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/simulation/simulation_server.hpp"

namespace gobot {
class InjectedStepWorld : public NullPhysicsWorld {
public:
    PhysicsStepResult next;
    PhysicsStepResult Step(RealType) override { return next; }
};

class PhysicsStepContract : public testing::Test {
protected:
    void SetUp() override {
        original = PhysicsServer::GetBackendInfo(PhysicsBackendType::Null);
        world = MakeRef<InjectedStepWorld>();
        ASSERT_TRUE(PhysicsServer::RegisterBackend(original, [this]() -> Ref<PhysicsWorld> { return world; }));
        root = Node::New<Node3D>();
        ASSERT_TRUE(simulation.BuildWorldFromScene(root));
        simulation.SetFixedTimeStep(0.01);
    }
    void TearDown() override {
        simulation.ClearWorld();
        Node::Delete(root);
        PhysicsServer::RegisterBackend(original, []() -> Ref<PhysicsWorld> { return MakeRef<NullPhysicsWorld>(); });
    }
    PhysicsBackendInfo original;
    Ref<InjectedStepWorld> world;
    Node3D* root{nullptr};
    SimulationServer simulation{PhysicsBackendType::Null, false};
};

TEST_F(PhysicsStepContract, RejectedStepDoesNotAdvanceClock) {
    world->next.error = "Injected rejection";
    EXPECT_FALSE(simulation.StepOnce());
    EXPECT_EQ(simulation.GetFrameCount(), 0);
    EXPECT_EQ(simulation.GetSimulationTime(), 0);
    EXPECT_TRUE(simulation.IsFaulted());
    EXPECT_TRUE(simulation.IsPaused());
    EXPECT_EQ(simulation.GetLastError(), "Injected rejection");
}

TEST_F(PhysicsStepContract, PartialStepReportsActualTimeAndLatchesFailure) {
    world->next = {.advanced_time = 0.004, .error = "Injected partial step"};
    EXPECT_FALSE(simulation.StepOnce());
    EXPECT_NEAR(simulation.GetSimulationTime(), 0.004, 1e-8);
    EXPECT_EQ(simulation.GetFrameCount(), 0);
    EXPECT_EQ(simulation.GetLastStepCount(), 0);
    EXPECT_TRUE(simulation.IsFaulted());
    EXPECT_FALSE(simulation.StepOnce());
    EXPECT_NEAR(simulation.GetSimulationTime(), 0.004, 1e-8);
    EXPECT_NEAR(simulation.GetLastPhysicsStepResult().advanced_time, 0.004, 1e-8);
    EXPECT_TRUE(simulation.Reset());
    EXPECT_EQ(simulation.GetSimulationTime(), 0);
    EXPECT_FALSE(simulation.IsFaulted());
}

TEST_F(PhysicsStepContract, InvalidAdvancementCannotPoisonClock) {
    world->next = {.completed = true, .advanced_time = std::numeric_limits<RealType>::quiet_NaN()};
    EXPECT_FALSE(simulation.StepOnce());
    EXPECT_EQ(simulation.GetSimulationTime(), 0);
    EXPECT_FALSE(simulation.GetLastPhysicsStepResult().state_valid);
}

TEST_F(PhysicsStepContract, CompletedStepAdvancesExactlyOnce) {
    world->next = {.completed = true, .advanced_time = 0.01};
    EXPECT_TRUE(simulation.StepOnce());
    EXPECT_NEAR(simulation.GetSimulationTime(), 0.01, 1e-8);
    EXPECT_EQ(simulation.GetFrameCount(), 1);
}
} // namespace gobot
