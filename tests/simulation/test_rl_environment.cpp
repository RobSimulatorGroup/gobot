#include <gtest/gtest.h>

#include <cmath>

#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
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

gobot::Robot3D* CreateTwoJointRobotScene() {
    auto* robot = CreateRobotScene();
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(robot->GetChild(1));
    auto* tip_link = joint == nullptr ? nullptr : gobot::Object::PointerCastTo<gobot::Link3D>(joint->GetChild(0));
    if (tip_link == nullptr) {
        return robot;
    }

    auto* second_joint = gobot::Object::New<gobot::Joint3D>();
    second_joint->SetName("second_joint");
    second_joint->SetJointType(gobot::JointType::Revolute);
    second_joint->SetParentLink("tip");
    second_joint->SetChildLink("foot");
    second_joint->SetLowerLimit(-2.0);
    second_joint->SetUpperLimit(2.0);

    auto* foot_link = gobot::Object::New<gobot::Link3D>();
    foot_link->SetName("foot");

    tip_link->AddChild(second_joint);
    second_joint->AddChild(foot_link);
    return robot;
}

gobot::Robot3D* CreateRobotSceneWithBaseCollision() {
    auto* robot = CreateRobotScene();
    auto* base_link = gobot::Object::PointerCastTo<gobot::Link3D>(robot->GetChild(0));
    if (base_link == nullptr) {
        return robot;
    }

    auto* collision_shape = gobot::Object::New<gobot::CollisionShape3D>();
    collision_shape->SetName("base_collision");
    auto shape = gobot::MakeRef<gobot::BoxShape3D>();
    shape->SetSize({0.5, 0.5, 0.5});
    collision_shape->SetShape(shape);
    base_link->AddChild(collision_shape);
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

    const gobot::RLEnvironmentResetResult reset_result = environment.Reset(123);
    ASSERT_TRUE(reset_result.ok);
    EXPECT_EQ(reset_result.seed, 123);
    EXPECT_EQ(reset_result.frame_count, 0);
    EXPECT_NEAR(reset_result.simulation_time, 0.0, CMP_EPSILON);
    EXPECT_TRUE(reset_result.error.empty());
    ASSERT_EQ(reset_result.observation.size(), 16);
    ASSERT_TRUE(simulation_server.HasWorld());
    EXPECT_EQ(environment.GetActionSize(), 1);
    EXPECT_EQ(environment.GetObservationSize(), 16);
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"joint"});
    EXPECT_TRUE(environment.GetContactLinkNames().empty());

    const std::vector<gobot::RealType> observation = environment.GetObservation();
    ASSERT_EQ(observation.size(), 16);
    ExpectBaseObservationPrefix(observation, {1.0, 2.0, 3.0});
    EXPECT_NEAR(observation[13], 0.25, CMP_EPSILON);
    EXPECT_NEAR(observation[14], 0.0, CMP_EPSILON);
    EXPECT_NEAR(observation[15], 0.0, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, exposes_action_and_observation_specs) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    ASSERT_TRUE(environment.Reset().ok);

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
                      "joint/previous_action",
              }));
    ASSERT_EQ(observation_spec.lower_bounds.size(), 16);
    ASSERT_EQ(observation_spec.upper_bounds.size(), 16);
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
                      "normalized",
              }));

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, exposes_contact_links_for_physical_links_with_collision_shapes) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotSceneWithBaseCollision();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    ASSERT_TRUE(environment.Reset().ok);

    EXPECT_EQ(environment.GetContactLinkNames(), std::vector<std::string>{"base"});
    EXPECT_EQ(environment.GetObservationSize(), 17);

    const std::vector<gobot::RealType> observation = environment.GetObservation();
    ASSERT_EQ(observation.size(), 17);
    EXPECT_DOUBLE_EQ(observation[15], 0.0);
    EXPECT_DOUBLE_EQ(observation[16], 0.0);

    const gobot::RLVectorSpec observation_spec = environment.GetObservationSpec();
    ASSERT_EQ(observation_spec.names.size(), 17);
    EXPECT_EQ(observation_spec.names.back(), "base/contact");
    EXPECT_DOUBLE_EQ(observation_spec.lower_bounds.back(), 0.0);
    EXPECT_DOUBLE_EQ(observation_spec.upper_bounds.back(), 1.0);
    EXPECT_EQ(observation_spec.units.back(), "boolean");

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    ASSERT_EQ(result.observation.size(), 17);
    EXPECT_DOUBLE_EQ(result.observation[16], 0.0);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, floating_joint_child_link_is_used_as_base_observation) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateFloatingBaseRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    ASSERT_TRUE(environment.Reset().ok);
    EXPECT_EQ(environment.GetActionSize(), 1);
    EXPECT_EQ(environment.GetObservationSize(), 16);
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

    ASSERT_TRUE(environment.Reset().ok);
    ASSERT_EQ(simulation_server.GetWorld()->GetSceneState().robots[0].joints.size(), 2);
    EXPECT_EQ(environment.GetActionSize(), 1);
    EXPECT_EQ(environment.GetObservationSize(), 16);
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"joint"});

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_EQ(result.observation.size(), 16);
    EXPECT_TRUE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, configured_controlled_joints_define_action_order_and_size) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateTwoJointRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    environment.SetConfiguredControlledJointNames({"second_joint"});

    ASSERT_TRUE(environment.Reset().ok);
    EXPECT_EQ(environment.GetControlledJointNames(), std::vector<std::string>{"second_joint"});
    EXPECT_EQ(environment.GetActionSize(), 1);
    EXPECT_EQ(environment.GetObservationSize(), 16);
    EXPECT_EQ(environment.GetActionSpec().names,
              std::vector<std::string>{"second_joint/target_position_normalized"});

    const gobot::RLEnvironmentStepResult result = environment.Step({0.5});
    EXPECT_TRUE(result.error.empty());
    ASSERT_EQ(result.observation.size(), 16);

    const auto& joint_states = simulation_server.GetWorld()->GetSceneState().robots[0].joints;
    ASSERT_EQ(joint_states.size(), 2);
    EXPECT_EQ(joint_states[0].control_mode, gobot::PhysicsJointControlMode::Passive);
    EXPECT_EQ(joint_states[1].control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(joint_states[1].target_position, 1.0);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, default_action_is_applied_on_reset) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateTwoJointRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    environment.SetConfiguredControlledJointNames({"second_joint"});
    environment.SetDefaultAction({-0.5});

    ASSERT_TRUE(environment.Reset().ok);
    EXPECT_EQ(environment.GetDefaultAction(), std::vector<gobot::RealType>{-0.5});

    const auto& joint_states = simulation_server.GetWorld()->GetSceneState().robots[0].joints;
    ASSERT_EQ(joint_states.size(), 2);
    EXPECT_EQ(joint_states[0].control_mode, gobot::PhysicsJointControlMode::Passive);
    EXPECT_EQ(joint_states[1].control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(joint_states[1].target_position, -1.0);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, invalid_default_action_size_fails_reset) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    environment.SetDefaultAction({0.0, 0.0});

    const gobot::RLEnvironmentResetResult result = environment.Reset();
    EXPECT_FALSE(result.ok);
    EXPECT_FALSE(result.error.empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, invalid_configured_controlled_joint_fails_reset) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotSceneWithFixedJoint();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    environment.SetConfiguredControlledJointNames({"fixed_joint"});

    const gobot::RLEnvironmentResetResult result = environment.Reset();
    EXPECT_FALSE(result.ok);
    EXPECT_FALSE(result.error.empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, reset_replays_deterministically_for_same_seed_and_actions) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.01);
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    ASSERT_TRUE(environment.Reset(7).ok);
    const std::vector<gobot::RealType> initial_observation = environment.GetObservation();
    std::vector<std::vector<gobot::RealType>> first_observations;
    for (const gobot::RealType action : {-1.0, 0.25, 1.0}) {
        const gobot::RLEnvironmentStepResult result = environment.Step({action});
        first_observations.push_back(result.observation);
    }

    ASSERT_TRUE(environment.Reset(7).ok);
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

    ASSERT_TRUE(environment.Reset().ok);

    const gobot::RLEnvironmentStepResult result = environment.Step({1.0});
    EXPECT_EQ(result.frame_count, 1);
    EXPECT_NEAR(result.simulation_time, 0.02, CMP_EPSILON);
    EXPECT_TRUE(result.truncated);
    EXPECT_FALSE(result.terminated);
    EXPECT_DOUBLE_EQ(result.reward, 1.0);
    ASSERT_EQ(result.observation.size(), 16);
    EXPECT_DOUBLE_EQ(result.observation[15], 1.0);

    const gobot::PhysicsJointState& joint_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(joint_state.control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(joint_state.target_position, 1.0);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, reward_can_penalize_action_rate) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    gobot::RLEnvironmentRewardSettings settings;
    settings.action_rate_penalty_scale = 0.25;
    environment.SetRewardSettings(settings);

    ASSERT_TRUE(environment.Reset().ok);

    const gobot::RLEnvironmentStepResult result = environment.Step({1.0});
    ASSERT_TRUE(result.error.empty());
    EXPECT_DOUBLE_EQ(result.reward, 0.75);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, terminates_when_base_height_is_too_low) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    gobot::RLEnvironmentRewardSettings settings;
    settings.minimum_base_height = 3.5;
    environment.SetRewardSettings(settings);

    ASSERT_TRUE(environment.Reset().ok);

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_TRUE(result.terminated);
    EXPECT_FALSE(result.truncated);
    EXPECT_DOUBLE_EQ(result.reward, settings.fallen_reward);
    EXPECT_TRUE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, terminates_when_base_tilt_is_too_large) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();
    auto* base_link = dynamic_cast<gobot::Link3D*>(robot->GetChild(0));
    ASSERT_NE(base_link, nullptr);
    base_link->SetEuler({2.0, 0.0, 0.0});

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    gobot::RLEnvironmentRewardSettings settings;
    settings.maximum_base_tilt_radians = 1.0;
    environment.SetRewardSettings(settings);

    ASSERT_TRUE(environment.Reset().ok);

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_TRUE(result.terminated);
    EXPECT_FALSE(result.truncated);
    EXPECT_DOUBLE_EQ(result.reward, settings.fallen_reward);
    EXPECT_TRUE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, can_report_fallen_reward_without_terminating) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");

    gobot::RLEnvironmentRewardSettings settings;
    settings.minimum_base_height = 3.5;
    settings.terminate_on_fall = false;
    settings.fallen_reward = -2.0;
    environment.SetRewardSettings(settings);

    ASSERT_TRUE(environment.Reset().ok);

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_FALSE(result.terminated);
    EXPECT_DOUBLE_EQ(result.reward, -2.0);

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, rejects_wrong_action_size_before_stepping) {
    gobot::SimulationServer simulation_server;
    auto* robot = CreateRobotScene();

    gobot::RLEnvironment environment(&simulation_server);
    environment.SetSceneRoot(robot);
    environment.SetRobotName("robot");
    ASSERT_TRUE(environment.Reset().ok);

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0, 1.0});
    EXPECT_TRUE(result.observation.empty());
    EXPECT_FALSE(result.error.empty());
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

    const gobot::RLEnvironmentResetResult reset_result = environment.Reset();
    EXPECT_FALSE(reset_result.ok);
    EXPECT_TRUE(reset_result.observation.empty());
    EXPECT_FALSE(reset_result.error.empty());
    EXPECT_FALSE(environment.GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestRLEnvironment, reports_error_when_stepping_before_reset) {
    gobot::SimulationServer simulation_server;
    gobot::RLEnvironment environment(&simulation_server);
    environment.SetRobotName("robot");

    const gobot::RLEnvironmentStepResult result = environment.Step({0.0});
    EXPECT_TRUE(result.observation.empty());
    EXPECT_FALSE(result.error.empty());
    EXPECT_FALSE(environment.GetLastError().empty());
}
