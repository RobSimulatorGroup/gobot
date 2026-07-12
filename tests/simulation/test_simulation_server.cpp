#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/capsule_shape_3d.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/window.hpp>
#include <gobot/physics/backends/mujoco_physics_world.hpp>
#include <gobot/simulation/simulation_server.hpp>

namespace gobot {

class CountingPhysicsNode : public Node {
    GOBCLASS(CountingPhysicsNode, Node)

public:
    void NotificationCallBack(NotificationType notification) {
        if (notification != NotificationType::PhysicsProcess) {
            return;
        }
        ++physics_process_count;
        physics_process_deltas.push_back(GetPhysicsProcessDeltaTime());
    }

    int physics_process_count{0};
    std::vector<double> physics_process_deltas;
};

} // namespace gobot

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

gobot::Robot3D* CreateTwoJointRobotScene() {
    auto* robot = CreateRobotScene();
    auto* first_joint = gobot::Object::PointerCastTo<gobot::Joint3D>(robot->GetChild(1));
    auto* tip_link = first_joint == nullptr ? nullptr : gobot::Object::PointerCastTo<gobot::Link3D>(first_joint->GetChild(0));
    if (tip_link == nullptr) {
        return robot;
    }

    auto* second_joint = gobot::Object::New<gobot::Joint3D>();
    second_joint->SetName("second_joint");
    second_joint->SetJointType(gobot::JointType::Revolute);
    second_joint->SetParentLink("tip");
    second_joint->SetChildLink("foot");
    second_joint->SetAxis({0.0, 1.0, 0.0});
    second_joint->SetLowerLimit(-2.0);
    second_joint->SetUpperLimit(2.0);

    auto* foot_link = gobot::Object::New<gobot::Link3D>();
    foot_link->SetName("foot");

    tip_link->AddChild(second_joint);
    second_joint->AddChild(foot_link);
    return robot;
}

gobot::CollisionShape3D* CreateBoxCollision(const std::string& name,
                                            const gobot::Vector3& size) {
    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName(name);
    auto shape = gobot::MakeRef<gobot::BoxShape3D>();
    shape->SetSize(size);
    collision->SetShape(shape);
    return collision;
}

gobot::Affine3 ComputeTransformFromRoot(const gobot::Node3D* node) {
    if (node == nullptr) {
        return gobot::Affine3::Identity();
    }

    gobot::Affine3 transform = node->GetTransform();
    const gobot::Node* parent = node->GetParent();
    while (parent != nullptr) {
        if (auto* parent_3d = gobot::Object::PointerCastTo<gobot::Node3D>(parent)) {
            transform = parent_3d->GetTransform() * transform;
        }
        parent = parent->GetParent();
    }
    return transform;
}

gobot::Affine3 GetGlobalOrRootTransform(const gobot::Node3D* node) {
    if (node == nullptr) {
        return gobot::Affine3::Identity();
    }
    if (node->IsInsideTree()) {
        return node->GetGlobalTransform();
    }
    return ComputeTransformFromRoot(node);
}

gobot::Link3D* FindLinkByName(gobot::Node* node, const std::string& name) {
    if (node == nullptr) {
        return nullptr;
    }
    if (auto* link = gobot::Object::PointerCastTo<gobot::Link3D>(node);
        link != nullptr && link->GetName() == name) {
        return link;
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (auto* found = FindLinkByName(node->GetChild(static_cast<int>(i)), name)) {
            return found;
        }
    }
    return nullptr;
}

gobot::Robot3D* CreateOffsetHingePendulumScene() {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("pendulum");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetRole(gobot::LinkRole::VirtualRoot);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("hinge");
    joint->SetJointType(gobot::JointType::Continuous);
    joint->SetParentLink("base");
    joint->SetChildLink("pole");
    joint->SetAxis({0.0, 1.0, 0.0});
    joint->SetJointPosition(0.2);

    auto* pole = gobot::Object::New<gobot::Link3D>();
    pole->SetName("pole");
    pole->SetPosition({0.0, 0.0, 0.5});
    pole->SetHasInertial(true);
    pole->SetMass(0.1);
    pole->SetCenterOfMass({0.0, 0.0, 0.0});
    pole->SetInertiaDiagonal({0.003, 0.003, 0.0002});
    pole->AddChild(CreateBoxCollision("pole_collision", {0.05, 0.05, 1.0}));

    robot->AddChild(base);
    base->AddChild(joint);
    joint->AddChild(pole);
    return robot;
}

gobot::Robot3D* CreateActuatedLimitedHingeScene() {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("limited");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetRole(gobot::LinkRole::VirtualRoot);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("calf");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetAxis({0.0, 1.0, 0.0});
    joint->SetLowerLimit(-2.818);
    joint->SetUpperLimit(-0.888);
    joint->SetJointPosition(-2.0);
    joint->SetInitialPosition(-2.0);
    joint->SetDriveMode(gobot::JointDriveMode::Position);
    joint->SetDriveStiffness(40.0);
    joint->SetControlLowerLimit(-2.818);
    joint->SetControlUpperLimit(-0.888);
    joint->SetForceLowerLimit(-40.0);
    joint->SetForceUpperLimit(40.0);

    auto* tip = gobot::Object::New<gobot::Link3D>();
    tip->SetName("tip");
    tip->SetPosition({0.0, 0.0, -0.25});
    tip->SetHasInertial(true);
    tip->SetMass(0.2);
    tip->SetInertiaDiagonal({0.002, 0.002, 0.002});

    robot->AddChild(base);
    base->AddChild(joint);
    joint->AddChild(tip);
    return robot;
}

gobot::Node3D* CreateSceneWithRobotAndGround(gobot::Robot3D* robot, gobot::CollisionShape3D** ground_out = nullptr) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("scene");

    auto* ground = CreateBoxCollision("ground", {4.0, 4.0, 0.1});
    ground->SetPosition({0.0, 0.0, -0.05});
    ground->SetFriction({1.0, 0.005, 0.0001});
    ground->SetContactType(3);
    ground->SetContactAffinity(5);
    ground->SetContactDimension(4);
    ground->SetSolref({0.012, 0.8});
    ground->SetSolimp({0.85, 0.94, 0.002, 0.45, 1.8});

    root->AddChild(ground);
    root->AddChild(robot);
    if (ground_out != nullptr) {
        *ground_out = ground;
    }
    return root;
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
    EXPECT_EQ(simulation_server.GetLastStepCount(), 2);
    EXPECT_EQ(simulation_server.GetFrameCount(), 2);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.2, CMP_EPSILON);
    EXPECT_NEAR(simulation_server.GetAccumulator(), 0.05, CMP_EPSILON);

    EXPECT_EQ(simulation_server.Step(0.05), 1);
    EXPECT_EQ(simulation_server.GetLastStepCount(), 1);
    EXPECT_EQ(simulation_server.GetFrameCount(), 3);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.3, CMP_EPSILON);
    EXPECT_NEAR(simulation_server.GetAccumulator(), 0.0, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, step_invokes_fixed_callback_before_each_world_step) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.002);
    simulation_server.SetMaxSubSteps(8);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    int callback_count = 0;
    std::vector<double> callback_deltas;
    std::vector<std::uint64_t> frame_counts_before_step;
    EXPECT_EQ(simulation_server.Step(0.016, [&](gobot::RealType fixed_delta) {
        ++callback_count;
        callback_deltas.push_back(static_cast<double>(fixed_delta));
        frame_counts_before_step.push_back(simulation_server.GetFrameCount());
    }), 8);

    EXPECT_EQ(callback_count, 8);
    ASSERT_EQ(callback_deltas.size(), 8);
    ASSERT_EQ(frame_counts_before_step.size(), 8);
    for (std::size_t index = 0; index < callback_deltas.size(); ++index) {
        EXPECT_NEAR(callback_deltas[index], 0.002, CMP_EPSILON);
        EXPECT_EQ(frame_counts_before_step[index], index);
    }
    EXPECT_EQ(simulation_server.GetFrameCount(), 8);
    EXPECT_NEAR(simulation_server.GetSimulationTime(), 0.016, CMP_EPSILON);
    EXPECT_NEAR(simulation_server.GetAccumulator(), 0.0, CMP_EPSILON);

    callback_count = 0;
    EXPECT_EQ(simulation_server.Step(0.001, [&](gobot::RealType) {
        ++callback_count;
    }), 0);
    EXPECT_EQ(callback_count, 0);
    EXPECT_EQ(simulation_server.GetFrameCount(), 8);
    EXPECT_NEAR(simulation_server.GetAccumulator(), 0.001, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, scene_tree_physics_notifications_follow_fixed_substeps_when_world_runs) {
    gobot::SceneTree tree(false);
    tree.Initialize();

    auto* counter = gobot::Object::New<gobot::CountingPhysicsNode>();
    counter->SetName("counter");
    tree.GetRoot()->AddChild(counter);

    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.002);
    simulation_server.SetMaxSubSteps(8);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateRobotScene();
    tree.GetRoot()->AddChild(robot);
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    tree.PhysicsProcess(0.016);

    EXPECT_EQ(counter->physics_process_count, 8);
    ASSERT_EQ(counter->physics_process_deltas.size(), 8);
    for (double delta : counter->physics_process_deltas) {
        EXPECT_NEAR(delta, 0.002, CMP_EPSILON);
    }
    EXPECT_EQ(simulation_server.GetFrameCount(), 8);

    tree.PhysicsProcess(0.001);
    EXPECT_EQ(counter->physics_process_count, 8);
    EXPECT_EQ(simulation_server.GetFrameCount(), 8);

    simulation_server.ClearWorld();
    tree.Finalize();
}

TEST(TestSimulationServer, paused_step_does_not_advance_but_step_once_does) {
    gobot::SimulationServer simulation_server;
    simulation_server.SetFixedTimeStep(0.02);

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    EXPECT_TRUE(simulation_server.IsPaused());

    EXPECT_EQ(simulation_server.Step(1.0), 0);
    EXPECT_EQ(simulation_server.GetLastStepCount(), 0);
    EXPECT_EQ(simulation_server.GetFrameCount(), 0);
    EXPECT_DOUBLE_EQ(simulation_server.GetSimulationTime(), 0.0);

    ASSERT_TRUE(simulation_server.StepOnce());
    EXPECT_EQ(simulation_server.GetLastStepCount(), 1);
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
    EXPECT_EQ(simulation_server.GetLastStepCount(), 0);
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

TEST(TestSimulationServer, can_defer_scene_sync_until_explicit_request) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(robot->GetChild(1));
    ASSERT_NE(joint, nullptr);

    robot->SetMode(gobot::RobotMode::Motion);
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    ASSERT_TRUE(simulation_server.GetWorld().IsValid());

    gobot::PhysicsSceneState moved_state = simulation_server.GetWorld()->GetSceneState();
    ASSERT_EQ(moved_state.robots.size(), 1);
    ASSERT_GE(moved_state.robots[0].joints.size(), 1);
    moved_state.robots[0].joints[0].position = 0.8;
    ASSERT_TRUE(simulation_server.GetWorld()->RestoreCompatibleState(moved_state));

    simulation_server.SetSyncSceneOnFixedStep(false);
    ASSERT_TRUE(simulation_server.StepOnce());
    EXPECT_DOUBLE_EQ(joint->GetJointPosition(), 0.25);

    ASSERT_TRUE(simulation_server.SyncSceneFromWorld());
    EXPECT_NEAR(joint->GetJointPosition(), 0.8, CMP_EPSILON);

    simulation_server.SetSyncSceneOnFixedStep(true);
    moved_state = simulation_server.GetWorld()->GetSceneState();
    moved_state.robots[0].joints[0].position = 0.2;
    ASSERT_TRUE(simulation_server.GetWorld()->RestoreCompatibleState(moved_state));
    ASSERT_TRUE(simulation_server.StepOnce());
    EXPECT_NEAR(joint->GetJointPosition(), 0.2, CMP_EPSILON);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, syncs_world_link_transform_to_motion_mode_robot) {
    gobot::SimulationServer simulation_server;

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({9.0, 9.0, 9.0});

    robot->AddChild(floating_joint);
    floating_joint->AddChild(base_link);
    ASSERT_NE(base_link, nullptr);

    robot->SetMode(gobot::RobotMode::Motion);

    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    ASSERT_TRUE(simulation_server.SyncSceneFromWorld());
    EXPECT_TRUE(base_link->GetTransform().translation().isApprox(gobot::Vector3(9.0, 9.0, 9.0), CMP_EPSILON));

    base_link->SetPosition({0.0, 0.0, 0.0});
    ASSERT_TRUE(simulation_server.Reset());
    EXPECT_TRUE(base_link->GetTransform().translation().isApprox(gobot::Vector3(9.0, 9.0, 9.0), CMP_EPSILON));

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, syncs_floating_joint_transform_to_motion_mode_robot) {
    gobot::SimulationServer simulation_server;

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    floating_joint->SetPosition({1.0, 2.0, 3.0});

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({0.0, 0.0, 0.5});

    robot->AddChild(floating_joint);
    floating_joint->AddChild(base_link);
    robot->SetMode(gobot::RobotMode::Motion);

    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    gobot::PhysicsSceneState moved_state = simulation_server.GetWorld()->GetSceneState();
    ASSERT_EQ(moved_state.robots.size(), 1);
    ASSERT_EQ(moved_state.robots[0].links.size(), 1);
    moved_state.robots[0].links[0].global_transform.translation() = gobot::Vector3(4.0, 5.0, 6.0);
    ASSERT_TRUE(simulation_server.GetWorld()->RestoreCompatibleState(moved_state));

    ASSERT_TRUE(simulation_server.SyncSceneFromWorld());
    EXPECT_TRUE(floating_joint->GetTransform().translation().isApprox(
            gobot::Vector3(4.0, 5.0, 5.5), CMP_EPSILON));
    EXPECT_TRUE(base_link->GetTransform().translation().isApprox(
            gobot::Vector3(0.0, 0.0, 0.5), CMP_EPSILON));
    EXPECT_TRUE((floating_joint->GetTransform() * base_link->GetTransform()).translation().isApprox(
            gobot::Vector3(4.0, 5.0, 6.0), CMP_EPSILON));

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, syncs_backend_link_pose_to_non_base_link_scene_transform) {
    gobot::SimulationServer simulation_server;
    auto* tree = gobot::SceneTree::New<gobot::SceneTree>(false);
    tree->Initialize();

    gobot::Robot3D* robot = CreateTwoJointRobotScene();
    auto* tip = FindLinkByName(robot, "tip");
    auto* foot = FindLinkByName(robot, "foot");
    ASSERT_NE(tip, nullptr);
    ASSERT_NE(foot, nullptr);
    auto* foot_collision = CreateBoxCollision("foot_collision", {0.1, 0.1, 0.1});
    foot->AddChild(foot_collision);
    tree->GetRoot()->AddChild(robot);
    robot->SetMode(gobot::RobotMode::Motion);

    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    gobot::PhysicsSceneState moved_state = simulation_server.GetWorld()->GetSceneState();
    ASSERT_EQ(moved_state.robots.size(), 1);
    ASSERT_GE(moved_state.robots[0].links.size(), 3);

    for (gobot::PhysicsLinkState& link_state : moved_state.robots[0].links) {
        if (link_state.link_name == "tip") {
            link_state.global_transform = gobot::Affine3::Identity();
            link_state.global_transform.translation() = gobot::Vector3(1.0, 2.0, 3.0);
        } else if (link_state.link_name == "foot") {
            link_state.global_transform = gobot::Affine3::Identity();
            link_state.global_transform.translation() = gobot::Vector3(4.0, 5.0, 6.0);
        }
    }
    ASSERT_TRUE(simulation_server.GetWorld()->RestoreCompatibleState(moved_state));
    ASSERT_TRUE(simulation_server.SyncSceneFromWorld());

    EXPECT_TRUE(ComputeTransformFromRoot(tip).translation().isApprox(gobot::Vector3(1.0, 2.0, 3.0),
                                                                      CMP_EPSILON));
    EXPECT_TRUE(GetGlobalOrRootTransform(foot).translation().isApprox(gobot::Vector3(4.0, 5.0, 6.0),
                                                                       CMP_EPSILON));
    EXPECT_TRUE(GetGlobalOrRootTransform(foot_collision).translation().isApprox(gobot::Vector3(4.0, 5.0, 6.0),
                                                                                CMP_EPSILON));

    tree->Finalize();
    gobot::SceneTree::Delete(tree);
}

TEST(TestSimulationServer, builds_runtime_scene_entities_with_base_articulation_and_control_types) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    const gobot::SimulationScene* runtime_scene = simulation_server.GetRuntimeScene();
    ASSERT_NE(runtime_scene, nullptr);
    ASSERT_EQ(runtime_scene->GetEntityCount(), 1);

    const gobot::SimulationEntity* entity = runtime_scene->GetEntity("robot");
    ASSERT_NE(entity, nullptr);
    EXPECT_TRUE(entity->IsFixedBase());
    EXPECT_FALSE(entity->IsFloatingBase());
    EXPECT_TRUE(entity->IsArticulated());
    EXPECT_TRUE(entity->IsActuated());
    EXPECT_TRUE(entity->HasJoint("joint"));
    EXPECT_TRUE(entity->HasLink("base"));
    EXPECT_TRUE(entity->HasControllableJoint("joint"));

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, reset_link_state_syncs_floating_base_to_motion_robot) {
    gobot::SimulationServer simulation_server;

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");

    robot->AddChild(floating_joint);
    floating_joint->AddChild(base_link);
    robot->SetMode(gobot::RobotMode::Motion);

    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    ASSERT_NE(simulation_server.GetRuntimeScene(), nullptr);
    ASSERT_TRUE(simulation_server.GetRuntimeScene()->ResetLinkState("robot",
                                                                    "base",
                                                                    gobot::Vector3(1.0, 2.0, 3.0),
                                                                    gobot::Quaternion::Identity(),
                                                                    gobot::Vector3(0.1, 0.2, 0.3),
                                                                    gobot::Vector3(0.4, 0.5, 0.6)));
    ASSERT_TRUE(simulation_server.SyncSceneFromWorld());

    const gobot::PhysicsLinkState& link_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].links[0];
    EXPECT_TRUE(link_state.global_transform.translation().isApprox(gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));
    EXPECT_TRUE(link_state.linear_velocity.isApprox(gobot::Vector3(0.1, 0.2, 0.3), CMP_EPSILON));
    EXPECT_TRUE(link_state.angular_velocity.isApprox(gobot::Vector3(0.4, 0.5, 0.6), CMP_EPSILON));
    EXPECT_TRUE(floating_joint->GetTransform().translation().isApprox(gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));

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

    ASSERT_NE(simulation_server.GetRuntimeScene(), nullptr);
    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetJointPositionTarget("robot", "joint", 0.5));
    const gobot::PhysicsJointState& position_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(position_state.control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(position_state.target_position, 0.5);

    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetJointVelocityTarget("robot", "joint", 1.25));
    const gobot::PhysicsJointState& velocity_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(velocity_state.control_mode, gobot::PhysicsJointControlMode::Velocity);
    EXPECT_DOUBLE_EQ(velocity_state.target_velocity, 1.25);

    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetJointEffortTarget("robot", "joint", 2.5));
    const gobot::PhysicsJointState& effort_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(effort_state.control_mode, gobot::PhysicsJointControlMode::Effort);
    EXPECT_DOUBLE_EQ(effort_state.target_effort, 2.5);

    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetJointPassive("robot", "joint"));
    const gobot::PhysicsJointState& passive_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(passive_state.control_mode, gobot::PhysicsJointControlMode::Passive);

    EXPECT_FALSE(simulation_server.GetRuntimeScene()->SetJointPositionTarget("robot", "missing", 0.0));
    EXPECT_FALSE(simulation_server.GetRuntimeScene()->GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, maps_normalized_robot_action_to_joint_position_targets) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    ASSERT_NE(simulation_server.GetRuntimeScene(), nullptr);
    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetRobotJointPositionTargetsFromNormalizedAction("robot", {-0.5}));
    const gobot::PhysicsJointState& joint_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(joint_state.control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(joint_state.target_position, -0.5);

    EXPECT_FALSE(simulation_server.GetRuntimeScene()->SetRobotJointPositionTargetsFromNormalizedAction("robot", {}));
    EXPECT_FALSE(simulation_server.GetRuntimeScene()->GetLastError().empty());
    EXPECT_FALSE(simulation_server.GetRuntimeScene()->SetRobotJointPositionTargetsFromNormalizedAction("robot", {0.0, 1.0}));
    EXPECT_FALSE(simulation_server.GetRuntimeScene()->GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, maps_named_normalized_robot_action_to_selected_joints) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateTwoJointRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    ASSERT_NE(simulation_server.GetRuntimeScene(), nullptr);

    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetRobotJointPositionTargetsFromNormalizedAction(
            "robot", std::vector<std::string>{"second_joint"}, std::vector<gobot::RealType>{0.5}));

    const auto& joint_states = simulation_server.GetWorld()->GetSceneState().robots[0].joints;
    ASSERT_EQ(joint_states.size(), 2);
    EXPECT_EQ(joint_states[0].control_mode, gobot::PhysicsJointControlMode::Passive);
    EXPECT_EQ(joint_states[1].control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(joint_states[1].target_position, 1.0);

    EXPECT_FALSE(simulation_server.GetRuntimeScene()->SetRobotJointPositionTargetsFromNormalizedAction(
            "robot", std::vector<std::string>{"missing"}, std::vector<gobot::RealType>{0.0}));
    EXPECT_FALSE(simulation_server.GetRuntimeScene()->GetLastError().empty());

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, default_joint_gains_update_existing_world_settings) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    gobot::JointControllerGains gains;
    gains.position_stiffness = 40.0;
    gains.velocity_damping = 4.0;
    gains.integral_gain = 1.5;
    gains.integral_limit = 0.25;
    simulation_server.SetDefaultJointGains(gains);

    const gobot::JointControllerGains& server_gains = simulation_server.GetDefaultJointGains();
    EXPECT_DOUBLE_EQ(server_gains.position_stiffness, 40.0);
    EXPECT_DOUBLE_EQ(server_gains.velocity_damping, 4.0);
    EXPECT_DOUBLE_EQ(server_gains.integral_gain, 1.5);
    EXPECT_DOUBLE_EQ(server_gains.integral_limit, 0.25);

    const gobot::JointControllerGains& world_gains =
            simulation_server.GetWorld()->GetSettings().default_joint_gains;
    EXPECT_DOUBLE_EQ(world_gains.position_stiffness, 40.0);
    EXPECT_DOUBLE_EQ(world_gains.velocity_damping, 4.0);
    EXPECT_DOUBLE_EQ(world_gains.integral_gain, 1.5);
    EXPECT_DOUBLE_EQ(world_gains.integral_limit, 0.25);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, mujoco_world_uses_training_solver_defaults) {
#ifdef GOBOT_HAS_MUJOCO
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);
    simulation_server.SetFixedTimeStep(0.002);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateActuatedLimitedHingeScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot)) << simulation_server.GetLastError();

    auto world = gobot::dynamic_pointer_cast<gobot::MuJoCoPhysicsWorld>(simulation_server.GetWorld());
    ASSERT_TRUE(world.IsValid());
    const gobot::MuJoCoPhysicsWorld::Diagnostics diagnostics = world->GetDiagnostics();
    EXPECT_NEAR(diagnostics.timestep, 0.002, CMP_EPSILON);
    EXPECT_EQ(diagnostics.solver, gobot::PhysicsSolverType::Newton);
    EXPECT_EQ(diagnostics.integrator, gobot::PhysicsIntegratorType::Euler);
    EXPECT_EQ(diagnostics.cone, gobot::PhysicsFrictionConeType::Pyramidal);
    EXPECT_EQ(diagnostics.jacobian, gobot::PhysicsJacobianType::Auto);
    EXPECT_EQ(diagnostics.iterations, 100);
    EXPECT_EQ(diagnostics.line_search_iterations, 50);
    EXPECT_EQ(diagnostics.no_slip_iterations, 0);
    EXPECT_EQ(diagnostics.convex_collision_iterations, 35);
    EXPECT_NEAR(diagnostics.tolerance, 1.0e-8, 1.0e-12);
    EXPECT_NEAR(diagnostics.line_search_tolerance, 0.01, 1.0e-8);
    EXPECT_NEAR(diagnostics.no_slip_tolerance, 1.0e-6, 1.0e-12);
    EXPECT_NEAR(diagnostics.convex_collision_tolerance, 1.0e-6, 1.0e-12);
    EXPECT_NEAR(diagnostics.impedance_ratio, 1.0, 1.0e-12);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestSimulationServer, mujoco_position_actuator_keeps_training_stiffness_and_damping) {
#ifdef GOBOT_HAS_MUJOCO
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);
    simulation_server.SetFixedTimeStep(0.002);
    simulation_server.SetPaused(false);
    gobot::JointControllerGains gains;
    gains.position_stiffness = 40.0;
    gains.velocity_damping = 1.0;
    simulation_server.SetDefaultJointGains(gains);

    gobot::Robot3D* robot = CreateActuatedLimitedHingeScene();
    ASSERT_EQ(robot->GetChildCount(), 1);
    auto* base = robot->GetChild(0);
    ASSERT_NE(base, nullptr);
    ASSERT_EQ(base->GetChildCount(), 1);
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(base->GetChild(0));
    ASSERT_NE(joint, nullptr);
    joint->SetDamping(1.0);

    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot)) << simulation_server.GetLastError();

    auto world = gobot::dynamic_pointer_cast<gobot::MuJoCoPhysicsWorld>(simulation_server.GetWorld());
    ASSERT_TRUE(world.IsValid());
    const gobot::MuJoCoPhysicsWorld::Diagnostics diagnostics = world->GetDiagnostics();
    EXPECT_EQ(diagnostics.actuator_count, 1);
    EXPECT_NEAR(diagnostics.first_position_actuator_stiffness, 40.0, 1.0e-9);
    EXPECT_NEAR(diagnostics.first_controllable_joint_damping, 1.0, 1.0e-9);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestSimulationServer, mujoco_authored_contact_parameters_match_scene_values) {
#ifdef GOBOT_HAS_MUJOCO
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);
    simulation_server.SetFixedTimeStep(0.002);
    simulation_server.SetPaused(false);

    gobot::Node3D* root = CreateSceneWithRobotAndGround(CreateActuatedLimitedHingeScene());
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(root)) << simulation_server.GetLastError();

    auto world = gobot::dynamic_pointer_cast<gobot::MuJoCoPhysicsWorld>(simulation_server.GetWorld());
    ASSERT_TRUE(world.IsValid());
    const gobot::MuJoCoPhysicsWorld::Diagnostics diagnostics = world->GetDiagnostics();
    EXPECT_NEAR(diagnostics.first_collision_friction.x(), 1.0, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_collision_friction.y(), 0.005, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_collision_friction.z(), 0.0001, CMP_EPSILON);
    EXPECT_EQ(diagnostics.first_collision_contact_dimension, 4);
    EXPECT_NEAR(diagnostics.first_collision_solref.x(), 0.012, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_collision_solref.y(), 0.8, CMP_EPSILON);
    ASSERT_EQ(diagnostics.first_collision_solimp.size(), 5);
    EXPECT_NEAR(diagnostics.first_collision_solimp[0], 0.85, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_collision_solimp[1], 0.94, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_collision_solimp[2], 0.002, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_collision_solimp[3], 0.45, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_collision_solimp[4], 1.8, CMP_EPSILON);

    gobot::Object::Delete(root);
#endif
}

TEST(TestSimulationServer, rebuild_world_preserves_compatible_joint_state_by_name) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));
    ASSERT_NE(simulation_server.GetRuntimeScene(), nullptr);
    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetJointPositionTarget("robot", "joint", 0.5));

    gobot::PhysicsJointState preserved_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    preserved_state.position = 0.42;
    preserved_state.velocity = 1.25;
    preserved_state.effort = 2.5;

    gobot::PhysicsSceneState previous_state = simulation_server.GetWorld()->GetSceneState();
    previous_state.robots[0].joints[0] = preserved_state;
    ASSERT_TRUE(simulation_server.GetWorld()->RestoreCompatibleState(previous_state));

    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(robot->GetChild(1));
    ASSERT_NE(joint, nullptr);
    joint->SetJointPosition(-0.25);

    ASSERT_TRUE(simulation_server.RebuildWorldFromScene(robot, true));
    const gobot::PhysicsJointState& rebuilt_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_NEAR(rebuilt_state.position, 0.42, CMP_EPSILON);
    EXPECT_DOUBLE_EQ(rebuilt_state.velocity, 1.25);
    EXPECT_DOUBLE_EQ(rebuilt_state.effort, 2.5);
    EXPECT_EQ(rebuilt_state.control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(rebuilt_state.target_position, 0.5);

    gobot::Object::Delete(robot);
}

TEST(TestSimulationServer, rebuild_world_does_not_preserve_incompatible_joint_type) {
    gobot::SimulationServer simulation_server;

    gobot::Robot3D* robot = CreateRobotScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot));

    gobot::PhysicsSceneState previous_state = simulation_server.GetWorld()->GetSceneState();
    previous_state.robots[0].joints[0].position = 0.75;
    ASSERT_TRUE(simulation_server.GetWorld()->RestoreCompatibleState(previous_state));

    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(robot->GetChild(1));
    ASSERT_NE(joint, nullptr);
    joint->SetJointType(gobot::JointType::Prismatic);
    joint->SetJointPosition(-0.25);

    ASSERT_TRUE(simulation_server.RebuildWorldFromScene(robot, true));
    const gobot::PhysicsJointState& rebuilt_state =
            simulation_server.GetWorld()->GetSceneState().robots[0].joints[0];
    EXPECT_DOUBLE_EQ(rebuilt_state.position, -0.25);
    EXPECT_EQ(rebuilt_state.joint_type, static_cast<int>(gobot::JointType::Prismatic));

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

TEST(TestSimulationServer, mujoco_does_not_load_robot_source_as_runtime_fallback) {
#ifdef GOBOT_HAS_MUJOCO
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);
    gobot::Robot3D* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("source_only_robot");
    robot->SetSourcePath("tests/fixtures/mjcf/simple_robot.xml");

    EXPECT_FALSE(simulation_server.BuildWorldFromScene(robot));
    EXPECT_FALSE(simulation_server.HasWorld());
    EXPECT_NE(simulation_server.GetLastError().find("no authored links or joints"), std::string::npos);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestSimulationServer, mujoco_authored_offset_hinge_pendulum_falls_under_gravity) {
#ifdef GOBOT_HAS_MUJOCO
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);
    simulation_server.SetFixedTimeStep(1.0 / 240.0);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateOffsetHingePendulumScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot)) << simulation_server.GetLastError();

    const auto& initial_joints = simulation_server.GetWorld()->GetSceneState().robots[0].joints;
    ASSERT_EQ(initial_joints.size(), 1);
    const gobot::RealType initial_position = initial_joints[0].position;

    for (int tick = 0; tick < 120; ++tick) {
        ASSERT_TRUE(simulation_server.StepOnce());
    }

    const auto& stepped_joints = simulation_server.GetWorld()->GetSceneState().robots[0].joints;
    ASSERT_EQ(stepped_joints.size(), 1);
    EXPECT_TRUE(std::isfinite(stepped_joints[0].position));
    EXPECT_GT(std::abs(stepped_joints[0].position - initial_position), 0.01);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestSimulationServer, mujoco_authored_position_actuator_respects_imported_limits) {
#ifdef GOBOT_HAS_MUJOCO
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);
    simulation_server.SetFixedTimeStep(1.0 / 240.0);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateActuatedLimitedHingeScene();
    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot)) << simulation_server.GetLastError();
    ASSERT_NE(simulation_server.GetRuntimeScene(), nullptr);
    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetJointPositionTarget("limited", "calf", 0.0));

    for (int tick = 0; tick < 240; ++tick) {
        ASSERT_TRUE(simulation_server.StepOnce()) << simulation_server.GetLastError();
    }

    const auto& joints = simulation_server.GetWorld()->GetSceneState().robots[0].joints;
    ASSERT_EQ(joints.size(), 1);
    EXPECT_TRUE(std::isfinite(joints[0].position));
    EXPECT_LE(joints[0].position, -0.888 + 1.0e-3);
    EXPECT_GT(joints[0].position, -2.0);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestSimulationServer, mujoco_position_actuator_without_control_range_allows_out_of_range_targets) {
#ifdef GOBOT_HAS_MUJOCO
    gobot::SimulationServer simulation_server(gobot::PhysicsBackendType::MuJoCoCpu);
    simulation_server.SetFixedTimeStep(1.0 / 240.0);
    simulation_server.SetPaused(false);

    gobot::Robot3D* robot = CreateActuatedLimitedHingeScene();
    ASSERT_EQ(robot->GetChildCount(), 1);
    auto* base = robot->GetChild(0);
    ASSERT_NE(base, nullptr);
    ASSERT_EQ(base->GetChildCount(), 1);
    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(base->GetChild(0));
    ASSERT_NE(joint, nullptr);
    joint->SetControlLowerLimit(0.0);
    joint->SetControlUpperLimit(0.0);

    ASSERT_TRUE(simulation_server.BuildWorldFromScene(robot)) << simulation_server.GetLastError();
    ASSERT_NE(simulation_server.GetRuntimeScene(), nullptr);
    ASSERT_TRUE(simulation_server.GetRuntimeScene()->SetJointPositionTarget("limited", "calf", 0.0));
    ASSERT_TRUE(simulation_server.StepOnce()) << simulation_server.GetLastError();

    auto world = gobot::dynamic_pointer_cast<gobot::MuJoCoPhysicsWorld>(simulation_server.GetWorld());
    ASSERT_TRUE(world.IsValid());
    const gobot::MuJoCoPhysicsWorld::Diagnostics diagnostics = world->GetDiagnostics();
    EXPECT_EQ(diagnostics.first_position_actuator_control_limited, 0);
    EXPECT_NEAR(diagnostics.first_position_actuator_control_value, 0.0, CMP_EPSILON);
    EXPECT_EQ(diagnostics.first_position_actuator_force_limited, 1);
    EXPECT_NEAR(diagnostics.first_position_actuator_force_range.x(), -40.0, CMP_EPSILON);
    EXPECT_NEAR(diagnostics.first_position_actuator_force_range.y(), 40.0, CMP_EPSILON);

    gobot::Object::Delete(robot);
#endif
}
