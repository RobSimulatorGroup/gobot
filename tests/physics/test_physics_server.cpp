#include <gtest/gtest.h>

#include <gobot/physics/physics_server.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/robot_3d.hpp>

TEST(TestPhysicsServer, exposes_backend_capabilities_without_optional_dependencies) {
    gobot::PhysicsServer physics_server;

    const gobot::PhysicsBackendInfo null_info =
            physics_server.GetBackendInfo(gobot::PhysicsBackendType::Null);
    EXPECT_TRUE(null_info.available);
    EXPECT_TRUE(null_info.cpu);
    EXPECT_FALSE(null_info.gpu);

    const gobot::PhysicsBackendInfo mujoco_info =
            physics_server.GetBackendInfo(gobot::PhysicsBackendType::MuJoCoCpu);
    EXPECT_TRUE(mujoco_info.cpu);
    EXPECT_FALSE(mujoco_info.gpu);
    EXPECT_TRUE(mujoco_info.robotics_focused);

#ifndef GOBOT_HAS_MUJOCO
    EXPECT_FALSE(mujoco_info.available);
    EXPECT_FALSE(mujoco_info.status.empty());
#endif

    EXPECT_EQ(physics_server.GetBackendInfos().size(), 5);
}

TEST(TestPhysicsServer, creates_world_for_selected_backend) {
    gobot::PhysicsServer physics_server;
    physics_server.SetBackendType(gobot::PhysicsBackendType::Null);

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 1.0 / 120.0;
    settings.gravity = {0.0, 0.0, -1.0};

    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(settings);
    ASSERT_TRUE(world.IsValid());
    EXPECT_EQ(world->GetBackendType(), gobot::PhysicsBackendType::Null);
    EXPECT_TRUE(world->IsAvailable());
    EXPECT_DOUBLE_EQ(world->GetSettings().fixed_time_step, settings.fixed_time_step);
    EXPECT_TRUE(world->GetSettings().gravity.isApprox(settings.gravity, CMP_EPSILON));
}

TEST(TestPhysicsServer, captures_robot_scene_snapshot) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    robot->SetSourcePath("res://robot.urdf");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base_link");
    base_link->SetMass(2.5);
    base_link->SetCenterOfMass({0.0, 0.0, 0.1});

    auto* collision_shape = gobot::Object::New<gobot::CollisionShape3D>();
    collision_shape->SetName("base_collision");
    auto box_shape = gobot::MakeRef<gobot::BoxShape3D>();
    box_shape->SetSize({1.0, 2.0, 3.0});
    collision_shape->SetShape(box_shape);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint1");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base_link");
    joint->SetChildLink("tip_link");
    joint->SetAxis({0.0, 0.0, 1.0});
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);
    joint->SetJointPosition(0.25);

    robot->AddChild(base_link);
    base_link->AddChild(collision_shape);
    robot->AddChild(joint);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1);
    EXPECT_EQ(snapshot.robots[0].name, "robot");
    EXPECT_EQ(snapshot.robots[0].source_path, "res://robot.urdf");
    ASSERT_EQ(snapshot.robots[0].links.size(), 1);
    EXPECT_EQ(snapshot.robots[0].links[0].name, "base_link");
    EXPECT_DOUBLE_EQ(snapshot.robots[0].links[0].mass, 2.5);
    ASSERT_EQ(snapshot.robots[0].links[0].collision_shapes.size(), 1);
    EXPECT_EQ(snapshot.robots[0].links[0].collision_shapes[0].type, gobot::PhysicsShapeType::Box);
    EXPECT_TRUE(snapshot.robots[0].links[0].collision_shapes[0].box_size.isApprox(
            gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));
    ASSERT_EQ(snapshot.robots[0].joints.size(), 1);
    EXPECT_EQ(snapshot.robots[0].joints[0].name, "joint1");
    EXPECT_EQ(snapshot.robots[0].joints[0].parent_link, "base_link");
    EXPECT_EQ(snapshot.robots[0].joints[0].child_link, "tip_link");
    EXPECT_DOUBLE_EQ(snapshot.robots[0].joints[0].joint_position, 0.25);
    EXPECT_EQ(snapshot.total_link_count, 1);
    EXPECT_EQ(snapshot.total_joint_count, 1);
    EXPECT_EQ(snapshot.total_collision_shape_count, 1);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, mujoco_world_reports_unavailable_when_not_built) {
    gobot::PhysicsServer physics_server(gobot::PhysicsBackendType::MuJoCoCpu);
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world.IsValid());
    EXPECT_EQ(world->GetBackendType(), gobot::PhysicsBackendType::MuJoCoCpu);

#ifndef GOBOT_HAS_MUJOCO
    EXPECT_FALSE(world->IsAvailable());
    EXPECT_FALSE(world->GetLastError().empty());
#endif
}
