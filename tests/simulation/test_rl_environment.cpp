#include <gtest/gtest.h>

#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/simulation/rl_environment.hpp>
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

gobot::Robot3D* CreateRobotSceneWithFixedJoint() {
    auto* robot = CreateRobotScene();

    auto* fixed_joint = gobot::Object::New<gobot::Joint3D>();
    fixed_joint->SetName("fixed_joint");
    fixed_joint->SetJointType(gobot::JointType::Fixed);
    fixed_joint->SetParentLink("tip");
    fixed_joint->SetChildLink("fixed_tip");

    auto* fixed_tip = gobot::Object::New<gobot::Link3D>();
    fixed_tip->SetName("fixed_tip");

    robot->AddChild(fixed_joint);
    fixed_joint->AddChild(fixed_tip);
    return robot;
}

} // namespace

TEST(TestRLEnvironment, reset_builds_world_and_returns_observation) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    ASSERT_TRUE(environment.Reset(123));
    ASSERT_TRUE(simulation_server.HasWorld());
    EXPECT_EQ(environment.GetActionSize(), 1);
    EXPECT_EQ(environment.GetObservationSize(), 2);
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"joint"});

    const std::vector<gobot::RealType> observation = environment.GetObservation();
    ASSERT_EQ(observation.size(), 2);
    EXPECT_NEAR(observation[0], 0.25, CMP_EPSILON);
    EXPECT_NEAR(observation[1], 0.0, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, ignores_fixed_joints_for_actions_and_observations) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotSceneWithFixedJoint();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    ASSERT_TRUE(environment.Reset());
    ASSERT_EQ(simulation_server.GetWorld()->GetSceneState().robots[0].joints.size(), 2);
    EXPECT_EQ(environment.GetActionSize(), 1);
    EXPECT_EQ(environment.GetObservationSize(), 2);
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"joint"});

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_EQ(result.observation.size(), 2);
    EXPECT_TRUE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, step_applies_normalized_action_and_advances_once) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.02);

    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    environment.SetMaxEpisodeSteps(1);

    ASSERT_TRUE(environment.Reset());

    const gobot::RLEnvironmentStepResult result = environment.Step({1.0});
    EXPECT_EQ(result.frame_count, 1);
    EXPECT_NEAR(result.simulation_time, 0.02, CMP_EPSILON);
    EXPECT_TRUE(result.truncated);
    EXPECT_FALSE(result.terminated);
    ASSERT_EQ(result.observation.size(), 2);

    const gobot::PhysicsJointState& joint_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(joint_state.control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(joint_state.target_position, 1.0);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, rejects_wrong_action_size_before_stepping) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    ASSERT_TRUE(environment.Reset());

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0, 1.0});
    EXPECT_TRUE(result.observation.empty());
    EXPECT_EQ(simulation_server.GetFrameCount(), 0);
    EXPECT_FALSE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, reset_fails_when_robot_name_is_missing) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("missing");

    EXPECT_FALSE(environment.Reset());
    EXPECT_FALSE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, reports_error_when_stepping_before_reset) {
    gobot::SimulationServer simulation_server;
    gobot::RLEnvironment environment(&simulation_server);
    environment.SetRobotName("robot");

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_TRUE(result.observation.empty());
    EXPECT_FALSE(environment.GetLastError().empty());
}
