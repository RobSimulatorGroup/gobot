#include <gtest/gtest.h>

#include <cmath>

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
    base_link->SetPosition({1.0, 2.0, 3.0});
    base_link->SetHasInertial(true);
    base_link->SetMass(1.0);

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

gobot::Robot3D* CreateFloatingBaseRobotScene() {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* virtual_root = gobot::Object::New<gobot::Link3D>();
    virtual_root->SetName("world");
    virtual_root->SetRole(gobot::LinkRole::VirtualRoot);
    virtual_root->SetPosition({9.0, 9.0, 9.0});

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetParentLink("world");
    floating_joint->SetChildLink("pelvis");

    auto* pelvis = gobot::Object::New<gobot::Link3D>();
    pelvis->SetName("pelvis");
    pelvis->SetPosition({0.0, 0.0, 0.75});

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("hip");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("pelvis");
    joint->SetChildLink("tip");
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);

    auto* tip_link = gobot::Object::New<gobot::Link3D>();
    tip_link->SetName("tip");

    robot->AddChild(virtual_root);
    virtual_root->AddChild(floating_joint);
    floating_joint->AddChild(pelvis);
    pelvis->AddChild(joint);
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

void ExpectBaseObservationPrefix(const std::vector<gobot::RealType>& observation,
                                 const gobot::Vector3& position) {
    ASSERT_GE(observation.size(), 13);
    EXPECT_NEAR(observation[0], position.x(), CMP_EPSILON);
    EXPECT_NEAR(observation[1], position.y(), CMP_EPSILON);
    EXPECT_NEAR(observation[2], position.z(), CMP_EPSILON);
    EXPECT_NEAR(observation[3], 0.0, CMP_EPSILON);
    EXPECT_NEAR(observation[4], 0.0, CMP_EPSILON);
    EXPECT_NEAR(observation[5], 0.0, CMP_EPSILON);
    EXPECT_NEAR(observation[6], 1.0, CMP_EPSILON);
    for (std::size_t index = 7; index < 13; ++index) {
        EXPECT_NEAR(observation[index], 0.0, CMP_EPSILON);
    }
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
    EXPECT_EQ(environment.GetObservationSize(), 15);
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"joint"});

    const std::vector<gobot::RealType> observation = environment.GetObservation();
    ASSERT_EQ(observation.size(), 15);
    ExpectBaseObservationPrefix(observation, {1.0, 2.0, 3.0});
    EXPECT_NEAR(observation[13], 0.25, CMP_EPSILON);
    EXPECT_NEAR(observation[14], 0.0, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, exposes_action_and_observation_specs) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    ASSERT_TRUE(environment.Reset());

    const gobot::RLVectorSpec action_spec = environment.GetActionSpec();
    EXPECT_EQ(action_spec.version, "rl_vector_spec_v1");
    EXPECT_EQ(action_spec.names, std::vector<std::string>{"joint/target_position_normalized"});
    EXPECT_EQ(action_spec.lower_bounds, std::vector<gobot::RealType>{-1.0});
    EXPECT_EQ(action_spec.upper_bounds, std::vector<gobot::RealType>{1.0});
    EXPECT_EQ(action_spec.units, std::vector<std::string>{"normalized"});

    const gobot::RLVectorSpec observation_spec = environment.GetObservationSpec();
    EXPECT_EQ(observation_spec.version, "rl_vector_spec_v1");
    EXPECT_EQ(observation_spec.names,
              std::vector<std::string>({
                      "base/position/x",
                      "base/position/y",
                      "base/position/z",
                      "base/orientation/x",
                      "base/orientation/y",
                      "base/orientation/z",
                      "base/orientation/w",
                      "base/linear_velocity/x",
                      "base/linear_velocity/y",
                      "base/linear_velocity/z",
                      "base/angular_velocity/x",
                      "base/angular_velocity/y",
                      "base/angular_velocity/z",
                      "joint/position",
                      "joint/velocity",
              }));
    ASSERT_EQ(observation_spec.lower_bounds.size(), 15);
    ASSERT_EQ(observation_spec.upper_bounds.size(), 15);
    EXPECT_TRUE(std::isinf(observation_spec.lower_bounds[0]));
    EXPECT_LT(observation_spec.lower_bounds[0], 0.0);
    EXPECT_TRUE(std::isinf(observation_spec.upper_bounds[0]));
    EXPECT_GT(observation_spec.upper_bounds[0], 0.0);
    EXPECT_DOUBLE_EQ(observation_spec.lower_bounds[3], -1.0);
    EXPECT_DOUBLE_EQ(observation_spec.upper_bounds[6], 1.0);
    EXPECT_EQ(observation_spec.units,
              std::vector<std::string>({
                      "meter",
                      "meter",
                      "meter",
                      "quaternion",
                      "quaternion",
                      "quaternion",
                      "quaternion",
                      "meter_per_second",
                      "meter_per_second",
                      "meter_per_second",
                      "radian_per_second",
                      "radian_per_second",
                      "radian_per_second",
                      "radian_or_meter",
                      "radian_per_second_or_meter_per_second",
              }));

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, floating_joint_child_link_is_used_as_base_observation) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateFloatingBaseRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    ASSERT_TRUE(environment.Reset());
    EXPECT_EQ(environment.GetActionSize(), 1);
    EXPECT_EQ(environment.GetObservationSize(), 15);
    ExpectBaseObservationPrefix(environment.GetObservation(), {9.0, 9.0, 9.75});
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"hip"});

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
    EXPECT_EQ(environment.GetObservationSize(), 15);
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"joint"});

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_EQ(result.observation.size(), 15);
    EXPECT_TRUE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, reset_replays_deterministically_for_same_seed_and_actions) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.01);
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    ASSERT_TRUE(environment.Reset(7));
    const std::vector<gobot::RealType> initial_observation = environment.GetObservation();
    std::vector<std::vector<gobot::RealType>> first_observations;
    for (const gobot::RealType action : {-1.0, 0.25, 1.0}) {
        const gobot::RLEnvironmentStepResult result = environment.Step({action});
        first_observations.push_back(result.observation);
    }

    ASSERT_TRUE(environment.Reset(7));
    EXPECT_EQ(simulation_server.GetFrameCount(), 0);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.0, CMP_EPSILON);
    EXPECT_EQ(environment.GetObservation(), initial_observation);

    std::vector<std::vector<gobot::RealType>> second_observations;
    for (const gobot::RealType action : {-1.0, 0.25, 1.0}) {
        const gobot::RLEnvironmentStepResult result = environment.Step({action});
        second_observations.push_back(result.observation);
    }

    EXPECT_EQ(second_observations, first_observations);
    EXPECT_EQ(simulation_server.GetFrameCount(), 3);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.03, CMP_EPSILON);

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
    ASSERT_EQ(result.observation.size(), 15);

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
