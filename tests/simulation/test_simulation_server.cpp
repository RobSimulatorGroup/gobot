#include <gtest/gtest.h>

#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/simulation/simulation_server.hpp>

namespace {

gobot::Robot3D* CreateRobotScene() {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetAxis({0.0, 0.0, 1.0});
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);
    joint->SetJointPosition(0.25);

    auto* tip_link = gobot::Object::New<gobot::Link3D>();
    tip_link->SetName("tip");

    robot->AddChild(base_link);
    robot->AddChild(joint);
    joint->AddChild(tip_link);
    return robot;
}

} // namespace

TEST(TestSimulationServer, builds_world_from_scene_and_steps_with_fixed_time_step) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.1);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    ASSERT_TRUE(simulation_server.HasWorld());
    ASSERT_TRUE(simulation_server.GetWorld().IsValid());
    EXPECT_EQ(simulation_server.GetWorld()->GetBackendType(), gobot::PhysicsBackendType::Null);
    EXPECT_EQ(simulation_server.GetFrameCount(), 0);
    EXPECT_DOUBLE_EQ(simulation_server.GetSimulationTime(), 0.0);

    EXPECT_EQ(simulation_server.Step(0.25), 2);
    EXPECT_EQ(simulation_server.GetFrameCount(), 2);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.2, CMP_EPSILON);
    EXPECT_NEAR(simulation_server.GetAccumulator(), 0.05, CMP_EPSILON);

    EXPECT_EQ(simulation_server.Step(0.05), 1);
    EXPECT_EQ(simulation_server.GetFrameCount(), 3);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.3, CMP_EPSILON);
    EXPECT_NEAR(simulation_server.GetAccumulator(), 0.0, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, paused_step_does_not_advance_but_step_once_does) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.02);

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    EXPECT_TRUE(simulation_server.IsPaused());

    EXPECT_EQ(simulation_server.Step(1.0), 0);
    EXPECT_EQ(simulation_server.GetFrameCount(), 0);
    EXPECT_DOUBLE_EQ(simulation_server.GetSimulationTime(), 0.0);

    ASSERT_TRUE(simulation_server.StepOnce());
    EXPECT_EQ(simulation_server.GetFrameCount(), 1);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.02, CMP_EPSILON);
    EXPECT_NEAR(simulation_server.GetAccumulator(), 0.0, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, reset_clears_clock_without_rebuilding_world) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.1);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    EXPECT_EQ(simulation_server.Step(0.3), 3);
    EXPECT_EQ(simulation_server.GetFrameCount(), 3);

    ASSERT_TRUE(simulation_server.Reset());
    EXPECT_TRUE(simulation_server.HasWorld());
    EXPECT_EQ(simulation_server.GetFrameCount(), 0);
    EXPECT_DOUBLE_EQ(simulation_server.GetSimulationTime(), 0.0);
    EXPECT_DOUBLE_EQ(simulation_server.GetAccumulator(), 0.0);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, syncs_world_joint_state_to_motion_mode_robot) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(robot->GetChild(1));
    ASSERT_NE(joint, nullptr);

    robot->SetMode(gobot::RobotMode::Motion);
    joint->SetJointPosition(0.0);

    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    EXPECT_DOUBLE_EQ(joint->GetJointPosition(), 0.0);

    ASSERT_TRUE(simulation_server.SyncSceneFromWorld());
    EXPECT_DOUBLE_EQ(joint->GetJointPosition(), 0.0);

    joint->SetJointPosition(0.75);
    ASSERT_TRUE(simulation_server.Reset());
    EXPECT_DOUBLE_EQ(joint->GetJointPosition(), 0.0);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, does_not_sync_world_joint_state_to_assembly_mode_robot) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(robot->GetChild(1));
    ASSERT_NE(joint, nullptr);

    robot->SetMode(gobot::RobotMode::Motion);
    joint->SetJointPosition(0.0);
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    robot->SetMode(gobot::RobotMode::Assembly);
    joint->SetJointPosition(0.75);
    ASSERT_TRUE(simulation_server.Reset());
    EXPECT_DOUBLE_EQ(joint->GetJointPosition(), 0.75);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, sets_joint_control_targets_on_world) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    ASSERT_TRUE(simulation_server.SetJointPositionTarget("robot", "joint", 0.5));
    const gobot::PhysicsJointState& position_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(position_state.control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(position_state.target_position, 0.5);

    ASSERT_TRUE(simulation_server.SetJointVelocityTarget("robot", "joint", 1.25));
    const gobot::PhysicsJointState& velocity_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(velocity_state.control_mode, gobot::PhysicsJointControlMode::Velocity);
    EXPECT_DOUBLE_EQ(velocity_state.target_velocity, 1.25);

    ASSERT_TRUE(simulation_server.SetJointEffortTarget("robot", "joint", 2.5));
    const gobot::PhysicsJointState& effort_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(effort_state.control_mode, gobot::PhysicsJointControlMode::Effort);
    EXPECT_DOUBLE_EQ(effort_state.target_effort, 2.5);

    ASSERT_TRUE(simulation_server.SetJointPassive("robot", "joint"));
    const gobot::PhysicsJointState& passive_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(passive_state.control_mode, gobot::PhysicsJointControlMode::Passive);

    EXPECT_FALSE(simulation_server.SetJointPositionTarget("robot", "missing", 0.0));
    EXPECT_FALSE(simulation_server.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, reports_error_when_stepping_without_world) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetPaused(false);

    EXPECT_EQ(simulation_server.Step(0.1), 0);
    EXPECT_FALSE(simulation_server.GetLastError().empty());
    EXPECT_FALSE(simulation_server.Reset());
    EXPECT_FALSE(simulation_server.StepOnce());
}

TEST(TestSimulationServer, unavailable_backend_build_failure_does_not_keep_world) {
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);

    gobot::Robot3D* robot = CreateRobotScene();
    robot->SetSourcePath("res://missing.urdf");

#ifndef GOBOT_HAS_MUJOCO
    EXPECT_FALSE(simulation_server.BuildWorldFromScene(robot));
    EXPECT_FALSE(simulation_server.HasWorld());
    EXPECT_FALSE(simulation_server.GetLastError().empty());
#endif

    gobot::Object::Delete(robot);
}
