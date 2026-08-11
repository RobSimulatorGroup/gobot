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

TEST(TestJointController, velocity_control_computes_damping_effort) {
    gobot::JointController controller({0.0, 4.0, 0.0, 0.0});

    gobot::JointControllerState state;
    state.velocity = -0.5;

    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Velocity;
    command.target_velocity = 1.5;

    EXPECT_NEAR(controller.ComputeEffort(state, command, {}, 0.01), 8.0, CMP_EPSILON);
}

TEST(TestJointController, effort_control_passes_through_target_effort) {
    gobot::JointController controller({100.0, 10.0, 5.0, 1.0});

    gobot::JointControllerState state;
    state.position = 100.0;
    state.velocity = 100.0;

    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Effort;
    command.target_position = -100.0;
    command.target_velocity = -100.0;
    command.target_effort = 2.5;

    EXPECT_DOUBLE_EQ(controller.ComputeEffort(state, command, {}, 0.01), 2.5);
}

TEST(TestJointController, passive_control_outputs_zero_effort) {
    gobot::JointController controller({100.0, 10.0, 5.0, 1.0});

    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Passive;
    command.target_effort = 10.0;

    EXPECT_DOUBLE_EQ(controller.ComputeEffort({}, command, {}, 0.01), 0.0);
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

TEST(TestJointController, actuator_model_delays_and_filters_commands_with_telemetry) {
    gobot::JointController controller;
    gobot::JointActuatorModel model;
    model.command_delay_steps = 2;
    model.command_deadband = 0.2;
    model.command_slew_rate = 1.0;
    controller.SetActuatorModel(model);

    gobot::JointControllerState state;
    state.position = 0.1;
    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Position;
    command.target_position = 1.0;

    gobot::JointControllerCommand processed =
            controller.ProcessCommand(state, command, 0.1);
    EXPECT_DOUBLE_EQ(processed.target_position, state.position);
    EXPECT_TRUE(controller.GetTelemetry().delayed);

    command.target_position = 2.0;
    processed = controller.ProcessCommand(state, command, 0.1);
    EXPECT_DOUBLE_EQ(processed.target_position, state.position);

    command.target_position = 3.0;
    processed = controller.ProcessCommand(state, command, 0.1);
    EXPECT_NEAR(processed.target_position, 0.2, CMP_EPSILON);
    EXPECT_TRUE(controller.GetTelemetry().rate_limited);
    EXPECT_DOUBLE_EQ(controller.GetTelemetry().commanded_target, 3.0);
    EXPECT_NEAR(controller.GetTelemetry().applied_target, 0.2, CMP_EPSILON);

    controller.Reset();
    model.command_delay_steps = 0;
    model.command_slew_rate = 0.0;
    controller.SetActuatorModel(model);
    command.target_position = 0.15;
    processed = controller.ProcessCommand(state, command, 0.1);
    EXPECT_DOUBLE_EQ(processed.target_position, state.position);
    EXPECT_TRUE(controller.GetTelemetry().deadband_applied);
}

TEST(TestJointController, actuator_effort_envelope_and_runtime_state_are_replayable) {
    gobot::JointController controller;
    gobot::JointActuatorModel model;
    model.command_delay_steps = 1;
    model.strength_scale = 0.5;
    model.motor_velocity_limit = 10.0;
    model.motor_stall_effort = 8.0;
    controller.SetActuatorModel(model);

    gobot::JointControllerState state;
    state.velocity = 5.0;
    gobot::JointControllerLimits limits;
    limits.effort_limit = 100.0;
    EXPECT_DOUBLE_EQ(controller.ApplyEffortEnvelope(20.0, state, limits), 4.0);
    EXPECT_DOUBLE_EQ(controller.GetStrengthScaleAtVelocity(state.velocity), 0.25);
    EXPECT_TRUE(controller.GetTelemetry().effort_saturated);

    gobot::JointControllerCommand command;
    command.mode = gobot::PhysicsJointControlMode::Effort;
    command.target_effort = 3.0;
    controller.ProcessCommand({}, command, 0.01);
    const gobot::JointControllerRuntimeState checkpoint =
            controller.CaptureRuntimeState();

    gobot::JointController restored;
    restored.SetActuatorModel(model);
    restored.RestoreRuntimeState(checkpoint);
    const gobot::JointControllerCommand expected =
            controller.ProcessCommand({}, command, 0.01);
    const gobot::JointControllerCommand actual =
            restored.ProcessCommand({}, command, 0.01);
    EXPECT_DOUBLE_EQ(actual.target_effort, expected.target_effort);
    EXPECT_EQ(restored.GetTelemetry().delayed,
              controller.GetTelemetry().delayed);
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
