#include <gtest/gtest.h>
#include "gobot/simulation/simulation_session.hpp"

namespace gobot {
namespace {
class FakeExecutor : public SimulationExecutor {
public:
    int calls{0};
    int fail_at{0};
    int closed{0};
    bool poison{false};
    bool reset_fails{false};
    std::vector<std::size_t> reset_indices;
    std::vector<SimulationMicrostepResult> Advance(double dt) override {
        ++calls;
        if (poison) return {{true, dt}, {true, std::numeric_limits<double>::quiet_NaN()}};
        if (calls == fail_at) return {{.error = "rolled back", .requires_reset = false},
                                     {.error = "Newton limit", .failing_shard = 1}};
        return {{true, dt}, {true, dt}};
    }
    void Reset(const std::vector<std::size_t>& indices) override {
        if (reset_fails) throw std::runtime_error("reset failure");
        reset_indices = indices;
    }
    void Close() override { ++closed; }
};
}

TEST(SimulationSessionContract, CountsMicrostepsAndPreservesCommittedTimeOnFailure) {
    auto executor = std::make_shared<FakeExecutor>();
    SimulationSession session(executor, 2, .002, 2);
    executor->fail_at = 4;
    auto result = session.Step({3});
    EXPECT_FALSE(result.Completed());
    EXPECT_EQ(result.environments[0].completed_microsteps, 3);
    EXPECT_EQ(result.environments[0].completed_ticks, 1);
    EXPECT_DOUBLE_EQ(result.environments[0].advanced_time, .006);
    EXPECT_DOUBLE_EQ(session.GetClocks()[0].time, .006);
    EXPECT_EQ(result.environments[1].failing_shard, 1);
    EXPECT_THROW(session.Step(), std::logic_error);
    EXPECT_EQ(executor->calls, 4);
    session.Reset();
    EXPECT_TRUE(session.Step().Completed());
    EXPECT_EQ(session.GetClocks()[0].episode, 1);
    EXPECT_EQ(session.GetClocks()[0].tick, 1);
}

TEST(SimulationSessionContract, InvalidBatchResultCannotPartiallyCommitClocks) {
    auto executor = std::make_shared<FakeExecutor>();
    SimulationSession session(executor, 2, .002);
    executor->poison = true;
    auto result = session.Step();
    for (const auto& env : result.environments) EXPECT_FALSE(env.state_valid);
    for (const auto& clock : session.GetClocks()) EXPECT_EQ(clock.time, 0);
}

TEST(SimulationSessionContract, ResetFailedEnvironmentPreservesHealthyMicrostepPhase) {
    auto executor = std::make_shared<FakeExecutor>();
    SimulationSession session(executor, 2, .002, 3);
    executor->fail_at = 5;
    const auto failed = session.Step({3});
    ASSERT_FALSE(failed.Completed());
    EXPECT_FALSE(failed.environments[0].requires_reset);
    EXPECT_TRUE(failed.environments[1].requires_reset);
    EXPECT_FALSE(session.GetClocks()[0].faulted);
    EXPECT_TRUE(session.GetClocks()[1].faulted);
    session.Reset({1});
    executor->fail_at = 8;
    const auto resumed = session.Step();
    EXPECT_EQ(resumed.environments[0].completed_ticks, 1);
    EXPECT_EQ(resumed.environments[1].completed_ticks, 0);
    EXPECT_EQ(session.GetClocks()[0].microstep, 6);
    EXPECT_EQ(session.GetClocks()[0].tick, 2);
    EXPECT_NEAR(session.GetClocks()[0].time, .012, 1e-14);
    EXPECT_EQ(session.GetClocks()[1].microstep, 2);
    EXPECT_EQ(session.GetClocks()[1].tick, 0);
    EXPECT_DOUBLE_EQ(session.GetClocks()[1].time, .004);
}

TEST(SimulationSessionContract, ResetPreservesUnselectedEnvironmentAndValidatesBeforeMutation) {
    auto executor = std::make_shared<FakeExecutor>();
    SimulationSession session(executor, 2, .002);
    ASSERT_TRUE(session.Step({3}).Completed());
    session.Reset({1});
    EXPECT_EQ(session.GetClocks()[0].tick, 3);
    EXPECT_DOUBLE_EQ(session.GetClocks()[0].time, .006);
    EXPECT_EQ(session.GetClocks()[1].tick, 0);
    EXPECT_EQ(session.GetClocks()[1].episode, 1);
    EXPECT_THROW(session.Reset({0, 0}), std::invalid_argument);
    EXPECT_THROW(session.Reset({2}), std::invalid_argument);
    EXPECT_EQ(executor->reset_indices, std::vector<std::size_t>{1});
    executor->reset_fails = true;
    EXPECT_THROW(session.Reset({0}), std::runtime_error);
    EXPECT_DOUBLE_EQ(session.GetClocks()[0].time, .006);
    EXPECT_TRUE(session.GetClocks()[0].faulted);
}

TEST(SimulationSessionContract, CloseIsIdempotentAndRequestsAreValidated) {
    auto executor = std::make_shared<FakeExecutor>();
    {
        SimulationSession session(executor, 2, .002);
        EXPECT_THROW(session.Step({0}), std::invalid_argument);
        session.Close();
        session.Close();
        EXPECT_THROW(session.Step(), std::logic_error);
    }
    EXPECT_EQ(executor->closed, 1);
    EXPECT_EQ(executor->calls, 0);
}
} // namespace gobot
