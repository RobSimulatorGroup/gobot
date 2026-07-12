#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gobot/simulation/locomotion_batch_runtime.hpp"

namespace gobot {
namespace {

PhysicsRobotBatchStepResult ContactHistory(std::size_t environment_count,
                                           std::vector<std::uint8_t> history) {
    PhysicsRobotBatchStepResult state;
    state.environment_count = environment_count;
    state.contact_shape_group_names = {"foot_0"};
    state.contact_history_tick_count = 4;
    state.contact_shape_group_history = std::move(history);
    return state;
}

TEST(TestLocomotionBatchRuntime, ReportsAirTimePeakHeightAndFirstLandingOnce) {
    LocomotionBatchRuntime runtime(1, 1);
    const std::array<float, 1> foot_heights{0.12f};

    PhysicsRobotBatchStepResult airborne = ContactHistory(1, {0, 0, 0, 0});
    runtime.UpdateFootHistory(airborne, foot_heights, 0.02, 0.005);

    EXPECT_FLOAT_EQ(runtime.FootContacts()[0], 0.0f);
    EXPECT_NEAR(runtime.FootAirTimes()[0], 0.02f, 1.0e-6f);
    EXPECT_FLOAT_EQ(runtime.FootPeakHeights()[0], 0.12f);
    EXPECT_FLOAT_EQ(runtime.FirstContacts()[0], 0.0f);
    EXPECT_FLOAT_EQ(runtime.LandingForces()[0], 0.0f);

    runtime.ClearStepContacts(0);
    runtime.FootContactForces()[0] = 3.0f;
    runtime.FootContactForces()[1] = 4.0f;
    PhysicsRobotBatchStepResult landing = ContactHistory(1, {0, 0, 1, 1});
    runtime.UpdateFootHistory(landing, foot_heights, 0.02, 0.005);

    EXPECT_FLOAT_EQ(runtime.FootContacts()[0], 1.0f);
    EXPECT_FLOAT_EQ(runtime.FootAirTimes()[0], 0.0f);
    EXPECT_FLOAT_EQ(runtime.FootPeakHeights()[0], 0.12f);
    EXPECT_FLOAT_EQ(runtime.FirstContacts()[0], 1.0f);
    EXPECT_FLOAT_EQ(runtime.LandingForces()[0], 5.0f);

    runtime.ClearStepContacts(0);
    PhysicsRobotBatchStepResult persistent_contact = ContactHistory(1, {1, 1, 1, 1});
    runtime.UpdateFootHistory(persistent_contact, foot_heights, 0.02, 0.005);

    EXPECT_FLOAT_EQ(runtime.FootContacts()[0], 1.0f);
    EXPECT_FLOAT_EQ(runtime.FootPeakHeights()[0], 0.0f);
    EXPECT_FLOAT_EQ(runtime.FirstContacts()[0], 0.0f);
    EXPECT_FLOAT_EQ(runtime.LandingForces()[0], 0.0f);
}

TEST(TestLocomotionBatchRuntime, ResetClearsOnlySelectedEnvironmentHistory) {
    LocomotionBatchRuntime runtime(2, 1);
    PhysicsRobotBatchStepResult airborne = ContactHistory(2, {
            0, 0, 0, 0,
            0, 0, 0, 0,
    });
    runtime.UpdateFootHistory(airborne, std::array<float, 2>{0.1f, 0.2f}, 0.02, 0.005);

    runtime.ClearResetContacts(std::array<std::size_t, 1>{0});

    EXPECT_FLOAT_EQ(runtime.FootAirTimes()[0], 0.0f);
    EXPECT_FLOAT_EQ(runtime.FootPeakHeights()[0], 0.0f);
    EXPECT_NEAR(runtime.FootAirTimes()[1], 0.02f, 1.0e-6f);
    EXPECT_FLOAT_EQ(runtime.FootPeakHeights()[1], 0.2f);

    runtime.ClearStepContacts(0);
    runtime.ClearStepContacts(1);
    runtime.FootContactForces()[0] = 2.0f;
    PhysicsRobotBatchStepResult landing = ContactHistory(2, {
            0, 0, 1, 1,
            0, 0, 0, 0,
    });
    runtime.UpdateFootHistory(landing, std::array<float, 2>{0.08f, 0.25f}, 0.02, 0.005);

    EXPECT_FLOAT_EQ(runtime.FirstContacts()[0], 1.0f);
    EXPECT_FLOAT_EQ(runtime.LandingForces()[0], 2.0f);
    EXPECT_NEAR(runtime.FootAirTimes()[1], 0.04f, 1.0e-6f);
    EXPECT_FLOAT_EQ(runtime.FootPeakHeights()[1], 0.25f);
}

TEST(TestLocomotionBatchRuntime, EnforcesFixedBatchDimensions) {
    LocomotionBatchRuntime runtime(2, 4);

    EXPECT_EQ(runtime.GetEnvironmentCount(), 2);
    EXPECT_EQ(runtime.GetFootCount(), 4);
    EXPECT_EQ(runtime.CommandRuntime().GetEnvironmentCount(), 2);
    EXPECT_EQ(runtime.FootContacts().size(), 8);
    EXPECT_EQ(runtime.FootContactForces().size(), 24);
    EXPECT_THROW(runtime.ClearStepContacts(2), std::out_of_range);

    PhysicsRobotBatchStepResult wrong_environment_count;
    wrong_environment_count.environment_count = 1;
    EXPECT_THROW(
            runtime.UpdateFootHistory(
                    wrong_environment_count,
                    std::array<float, 8>{},
                    0.02,
                    0.005),
            std::invalid_argument);

    PhysicsRobotBatchStepResult malformed_history;
    malformed_history.environment_count = 2;
    malformed_history.contact_shape_group_names = {"foot_0"};
    malformed_history.contact_history_tick_count = 1;
    malformed_history.contact_shape_group_history = {1};
    EXPECT_THROW(
            runtime.UpdateFootHistory(
                    malformed_history,
                    std::array<float, 8>{},
                    0.02,
                    0.005),
            std::invalid_argument);
}

} // namespace
} // namespace gobot
