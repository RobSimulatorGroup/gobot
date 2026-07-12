#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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

LocomotionBatchStateLayout StateLayout() {
    LocomotionBatchStateLayout layout;
    layout.robot_name = "robot";
    layout.base_link = "base";
    layout.joint_names = {"joint"};
    layout.link_names = {"base", "foot"};
    layout.sensor_names = {"imu", "foot_height", "foot_contact", "height_scan"};
    layout.foot_link_indices = {1};
    layout.foot_shape_names = {"foot_collision"};
    layout.foot_height_sensor_indices = {1};
    layout.foot_contact_sensor_indices = {2};
    layout.height_scan_sensor_index = 3;
    layout.imu_sensor_index = 0;
    layout.height_scan_count = 2;
    return layout;
}

PhysicsRobotBatchStepResult PhysicsState() {
    PhysicsRobotBatchStepResult state;
    state.robot_name = "robot";
    state.base_link = "base";
    state.joint_names = {"joint"};
    state.link_names = {"base", "foot"};
    state.sensor_names = {"imu", "foot_height", "foot_contact", "height_scan"};
    state.shape_names = {"foot_collision"};
    state.environment_count = 1;
    state.max_sensor_values = 10;
    state.max_sensor_hits = 2;
    state.max_contact_count = 1;

    constexpr RealType kHalfSqrtTwo = 0.7071067811865476;
    state.base_position = {1.0, 2.0, 0.5};
    state.base_quaternion = {kHalfSqrtTwo, 0.0, kHalfSqrtTwo, 0.0};
    state.base_linear_velocity = {1.0, 2.0, 3.0};
    state.base_angular_velocity = {4.0, 5.0, 6.0};
    state.joint_position = {0.25};
    state.joint_velocity = {0.5};
    state.joint_acceleration = {0.75};
    state.joint_effort = {1.25};
    state.joint_lower_limit = {-1.0};
    state.joint_upper_limit = {1.0};
    state.link_position = {1.0, 2.0, 0.5, 0.0, 0.0, 0.1};
    state.link_quaternion = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    state.link_linear_velocity = {1.0, 2.0, 3.0, 0.1, 0.2, 0.3};
    state.link_angular_velocity = {4.0, 5.0, 6.0, 0.4, 0.5, 0.6};

    state.sensor_value_count = {10, 1, 0, 2};
    state.sensor_hit_count = {0, 0, 0, 2};
    state.sensor_position.assign(4 * 3, 0.0);
    state.sensor_position[2 * 3 + 0] = 0.2;
    state.sensor_position[2 * 3 + 1] = 0.3;
    state.sensor_position[2 * 3 + 2] = 0.4;
    state.sensor_quaternion.assign(4 * 4, 0.0);
    for (std::size_t sensor_index = 0; sensor_index < 4; ++sensor_index) {
        state.sensor_quaternion[sensor_index * 4] = 1.0;
    }
    state.sensor_linear_velocity.assign(4 * 3, 0.0);
    state.sensor_linear_velocity[2 * 3 + 0] = 0.7;
    state.sensor_linear_velocity[2 * 3 + 1] = 0.8;
    state.sensor_linear_velocity[2 * 3 + 2] = 0.9;
    state.sensor_values.assign(4 * 10, 0.0);
    state.sensor_values[4] = 7.0;
    state.sensor_values[5] = 8.0;
    state.sensor_values[6] = 9.0;
    state.sensor_values[7] = 10.0;
    state.sensor_values[8] = 11.0;
    state.sensor_values[9] = 12.0;
    state.sensor_values[10] = 0.08;
    state.sensor_values[30] = 0.2;
    state.sensor_values[31] = 0.4;
    state.sensor_hit.assign(4 * 2, 0);
    state.sensor_hit[6] = 1;
    state.sensor_hit_point.assign(4 * 2 * 3, 0.0);
    state.sensor_hit_point[6 * 3 + 0] = 1.0;
    state.sensor_hit_point[6 * 3 + 1] = 2.0;
    state.sensor_hit_point[6 * 3 + 2] = 3.0;
    state.sensor_hit_normal.assign(4 * 2 * 3, 0.0);
    state.sensor_hit_normal[6 * 3 + 2] = 1.0;

    state.contact_count = {1};
    state.contact_link_index = {1, -1};
    state.contact_shape_index = {0, -1};
    state.contact_force = {3.0, 4.0, 5.0};
    state.contact_shape_group_names = {
            "thigh", "shank", "trunk_head", "foot_0"};
    state.contact_shape_group_tick_count = {2, 3, 4, 1};
    state.self_contact_tick_count = {5};
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

TEST(TestLocomotionBatchRuntime, ExtractsRobotSensorAndContactState) {
    LocomotionBatchRuntime runtime(1, StateLayout());
    const PhysicsRobotBatchStepResult state = PhysicsState();

    runtime.UpdateState(state);

    EXPECT_TRUE(runtime.HasStateLayout());
    EXPECT_EQ(runtime.GetEnvironmentCount(), 1);
    EXPECT_EQ(runtime.GetJointCount(), 1);
    EXPECT_EQ(runtime.GetLinkCount(), 2);
    EXPECT_EQ(runtime.GetFootCount(), 1);
    EXPECT_EQ(runtime.GetHeightScanCount(), 2);
    EXPECT_FLOAT_EQ(runtime.BasePositions()[2], 0.5f);
    EXPECT_NEAR(runtime.ProjectedGravity()[0], 1.0f, 1.0e-6f);
    EXPECT_NEAR(runtime.ProjectedGravity()[1], 0.0f, 1.0e-6f);
    EXPECT_NEAR(runtime.ProjectedGravity()[2], 0.0f, 1.0e-6f);
    EXPECT_FLOAT_EQ(runtime.BaseAngularVelocitiesBody()[0], 7.0f);
    EXPECT_FLOAT_EQ(runtime.BaseAngularVelocitiesBody()[2], 9.0f);
    EXPECT_FLOAT_EQ(runtime.BaseLinearVelocitiesBody()[0], 10.0f);
    EXPECT_FLOAT_EQ(runtime.BaseLinearVelocitiesBody()[2], 12.0f);
    EXPECT_FLOAT_EQ(runtime.UpVectors()[2], 1.0f);
    EXPECT_FLOAT_EQ(runtime.JointPositions()[0], 0.25f);
    EXPECT_FLOAT_EQ(runtime.JointVelocities()[0], 0.5f);
    EXPECT_FLOAT_EQ(runtime.JointAccelerations()[0], 0.75f);
    EXPECT_FLOAT_EQ(runtime.JointTorques()[0], 1.25f);
    EXPECT_FLOAT_EQ(runtime.JointLowerLimits()[0], -1.0f);
    EXPECT_FLOAT_EQ(runtime.JointUpperLimits()[0], 1.0f);
    EXPECT_FLOAT_EQ(runtime.FootPositions()[0], 0.2f);
    EXPECT_FLOAT_EQ(runtime.FootPositions()[2], 0.4f);
    EXPECT_FLOAT_EQ(runtime.FootVelocities()[0], 0.7f);
    EXPECT_FLOAT_EQ(runtime.FootHeights()[0], 0.08f);
    EXPECT_FLOAT_EQ(runtime.HeightScan()[0], 0.2f);
    EXPECT_FLOAT_EQ(runtime.HeightScan()[1], 0.4f);
    EXPECT_EQ(runtime.HeightScanHits()[0], 1);
    EXPECT_EQ(runtime.HeightScanHits()[1], 0);
    EXPECT_FLOAT_EQ(runtime.HeightScanPoints()[2], 3.0f);
    EXPECT_FLOAT_EQ(runtime.HeightScanNormals()[2], 1.0f);
    EXPECT_FLOAT_EQ(runtime.FootContacts()[0], 1.0f);
    EXPECT_FLOAT_EQ(runtime.FootContactForces()[0], 3.0f);
    EXPECT_FLOAT_EQ(runtime.FootContactForces()[1], 4.0f);
    EXPECT_FLOAT_EQ(runtime.FootContactForces()[2], 5.0f);
    EXPECT_FLOAT_EQ(runtime.IllegalContactCounts()[0], 2.0f);
    EXPECT_FLOAT_EQ(runtime.SelfCollisionCounts()[0], 5.0f);
    EXPECT_FLOAT_EQ(runtime.ShankCollisionCounts()[0], 3.0f);
    EXPECT_FLOAT_EQ(runtime.TrunkHeadCollisionCounts()[0], 4.0f);
}

TEST(TestLocomotionBatchRuntime, RejectsStateOutsideResolvedLayout) {
    LocomotionBatchRuntime runtime(1, StateLayout());

    PhysicsRobotBatchStepResult wrong_names = PhysicsState();
    wrong_names.joint_names = {"other_joint"};
    EXPECT_THROW(runtime.UpdateState(wrong_names), std::invalid_argument);

    PhysicsRobotBatchStepResult wrong_shape = PhysicsState();
    wrong_shape.link_angular_velocity.pop_back();
    EXPECT_THROW(runtime.UpdateState(wrong_shape), std::invalid_argument);

    PhysicsRobotBatchStepResult short_scan = PhysicsState();
    short_scan.sensor_hit_count[3] = 1;
    EXPECT_THROW(runtime.UpdateState(short_scan), std::invalid_argument);

    LocomotionBatchRuntime contact_only(1, 1);
    EXPECT_THROW(contact_only.UpdateState(PhysicsState()), std::logic_error);
}

} // namespace
} // namespace gobot
