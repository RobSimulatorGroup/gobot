#include <gtest/gtest.h>

#include <gobot/physics/joint_controller.hpp>
#include <gobot/physics/physics_types.hpp>

TEST(TestJointController, position_control_computes_pd_effort) {
    gobot::JointController controller({10.0, 2.0, 0.0, 0.0});

    gobot::JointControllerState state;
    state.position = 0.25;
    state.velocity = 0.5;

    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Position;
    command.target_position = 1.0;

    gobot::JointControllerLimits limits;

    EXPECT_NEAR(controller.ComputeEffort(state, command, limits, 0.01), 6.5, CMP_EPSILON);
}

TEST(TestJointController, effort_is_saturated_when_limit_is_positive) {
    gobot::JointController controller({100.0, 0.0, 0.0, 0.0});

    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Position;
    command.target_position = 10.0;

    gobot::JointControllerLimits limits;
    limits.effort_limit = 3.0;

    EXPECT_DOUBLE_EQ(controller.ComputeEffort({}, command, limits, 0.01), 3.0);

    command.target_position = -10.0;
    EXPECT_DOUBLE_EQ(controller.ComputeEffort({}, command, limits, 0.01), -3.0);
}

TEST(TestJointController, integral_error_is_clamped_and_resettable) {
    gobot::JointController controller({0.0, 0.0, 5.0, 0.2});

    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Position;
    command.target_position = 1.0;

    EXPECT_NEAR(controller.ComputeEffort({}, command, {}, 1.0), 1.0, CMP_EPSILON);
    EXPECT_NEAR(controller.GetIntegralError(), 0.2, CMP_EPSILON);

    EXPECT_NEAR(controller.ComputeEffort({}, command, {}, 1.0), 1.0, CMP_EPSILON);
    EXPECT_NEAR(controller.GetIntegralError(), 0.2, CMP_EPSILON);

    controller.Reset();
    EXPECT_DOUBLE_EQ(controller.GetIntegralError(), 0.0);
}

TEST(TestJointController, normalized_action_maps_to_joint_limits) {
    gobot::JointControllerLimits limits;
    limits.has_position_limits = true;
    limits.lower_position_limit = -2.0;
    limits.upper_position_limit = 4.0;

    EXPECT_DOUBLE_EQ(gobot::JointController::MapNormalizedActionToTargetPosition(-1.0, limits), -2.0);
    EXPECT_DOUBLE_EQ(gobot::JointController::MapNormalizedActionToTargetPosition(0.0, limits), 1.0);
    EXPECT_DOUBLE_EQ(gobot::JointController::MapNormalizedActionToTargetPosition(1.0, limits), 4.0);
    EXPECT_DOUBLE_EQ(gobot::JointController::MapNormalizedActionToTargetPosition(2.0, limits), 4.0);
}

TEST(TestJointController, command_helpers_convert_physics_joint_state) {
    gobot::PhysicsJointState joint_state;
    joint_state.position = 0.25;
    joint_state.velocity = -0.5;
    joint_state.control_mode = gobot::PhysicsJointControlMode::Position;
    joint_state.target_position = 0.75;

    const gobot::JointControllerState state = gobot::MakeJointControllerState(joint_state);
    EXPECT_DOUBLE_EQ(state.position, 0.25);
    EXPECT_DOUBLE_EQ(state.velocity, -0.5);

    const gobot::JointControllerCommand command = gobot::MakeJointControllerCommand(joint_state);
    EXPECT_EQ(command.mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(command.target_position, 0.75);
}
