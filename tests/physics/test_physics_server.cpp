#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cmath>

#include <gobot/physics/physics_server.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/mesh.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/sensor_3d.hpp>
#include <gobot/scene/terrain_3d.hpp>

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

    const gobot::PhysicsBackendInfo mujoco_warp_info =
            physics_server.GetBackendInfo(gobot::PhysicsBackendType::MuJoCoWarp);
    EXPECT_FALSE(mujoco_warp_info.available);
    EXPECT_FALSE(mujoco_warp_info.cpu);
    EXPECT_TRUE(mujoco_warp_info.gpu);
    EXPECT_TRUE(mujoco_warp_info.robotics_focused);
    EXPECT_FALSE(mujoco_warp_info.status.empty());

    const gobot::PhysicsBackendInfo rigid_ipc_info =
            physics_server.GetBackendInfo(gobot::PhysicsBackendType::RigidIpcCpu);
    EXPECT_FALSE(rigid_ipc_info.available);
    EXPECT_TRUE(rigid_ipc_info.cpu);
    EXPECT_FALSE(rigid_ipc_info.gpu);
    EXPECT_FALSE(rigid_ipc_info.status.empty());

    EXPECT_EQ(physics_server.GetBackendInfos().size(), 5);
}

TEST(TestPhysicsServer, creates_world_for_selected_backend) {
    gobot::PhysicsServer physics_server;
    physics_server.SetBackendType(gobot::PhysicsBackendType::Null);

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 1.0 / 120.0;
    settings.gravity = {0.0, 0.0, -1.0};
    settings.default_joint_gains.position_stiffness = 25.0;
    settings.default_joint_gains.velocity_damping = 5.0;
    settings.default_joint_gains.integral_gain = 1.5;
    settings.default_joint_gains.integral_limit = 0.25;

    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(settings);
    ASSERT_TRUE(world.IsValid());
    EXPECT_EQ(world->GetBackendType(), gobot::PhysicsBackendType::Null);
    EXPECT_TRUE(world->IsAvailable());
    EXPECT_DOUBLE_EQ(world->GetSettings().fixed_time_step, settings.fixed_time_step);
    EXPECT_TRUE(world->GetSettings().gravity.isApprox(settings.gravity, CMP_EPSILON));
    EXPECT_DOUBLE_EQ(world->GetSettings().default_joint_gains.position_stiffness, 25.0);
    EXPECT_DOUBLE_EQ(world->GetSettings().default_joint_gains.velocity_damping, 5.0);
    EXPECT_DOUBLE_EQ(world->GetSettings().default_joint_gains.integral_gain, 1.5);
    EXPECT_DOUBLE_EQ(world->GetSettings().default_joint_gains.integral_limit, 0.25);
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
    EXPECT_EQ(snapshot.robots[0].links[0].role, gobot::PhysicsLinkRole::Physical);
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

TEST(TestPhysicsServer, initializes_scene_state_from_robot_snapshot) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint1");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);
    joint->SetJointPosition(0.5);

    robot->AddChild(joint);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    EXPECT_EQ(state.robots[0].name, "robot");
    EXPECT_EQ(state.total_link_count, 0);
    ASSERT_EQ(state.robots[0].joints.size(), 1);
    EXPECT_EQ(state.robots[0].joints[0].joint_name, "joint1");
    EXPECT_EQ(state.robots[0].joints[0].robot_name, "robot");
    EXPECT_DOUBLE_EQ(state.robots[0].joints[0].position, 0.5);
    EXPECT_EQ(state.total_joint_count, 1);

    joint->SetJointPosition(-0.5);
    world->Reset();
    EXPECT_DOUBLE_EQ(world->GetSceneState().robots[0].joints[0].position, 0.5);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, initializes_link_state_from_robot_snapshot) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({1.0, 2.0, 3.0});

    robot->AddChild(base_link);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    ASSERT_EQ(state.robots[0].links.size(), 1);
    EXPECT_EQ(state.robots[0].links[0].link_name, "base");
    EXPECT_EQ(state.robots[0].links[0].robot_name, "robot");
    EXPECT_TRUE(state.robots[0].links[0].global_transform.translation().isApprox(
            gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));
    EXPECT_EQ(state.total_link_count, 1);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, captures_sensor_nodes_in_snapshot_and_state) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base");
    base_link->SetPosition({1.0, 2.0, 3.0});

    auto* imu = gobot::Object::New<gobot::IMUSensor3D>();
    imu->SetName("imu");
    imu->SetPosition({0.1, 0.2, 0.3});
    imu->SetSensorPeriod(0.01);
    imu->SetNoiseStddev(0.02);
    base_link->AddChild(imu);

    auto* angular_momentum = gobot::Object::New<gobot::AngularMomentumSensor3D>();
    angular_momentum->SetName("root_angmom");
    angular_momentum->SetSensorPeriod(0.02);
    base_link->AddChild(angular_momentum);

    auto* contact = gobot::Object::New<gobot::ContactSensor3D>();
    contact->SetName("foot_contact");
    contact->SetPosition({0.0, 0.0, -0.2});
    contact->SetRadius(0.05);
    contact->SetMinThreshold(0.1);
    contact->SetMaxThreshold(100.0);
    base_link->AddChild(contact);
    robot->AddChild(base_link);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1);
    ASSERT_EQ(snapshot.robots[0].sensors.size(), 3);
    EXPECT_EQ(snapshot.total_sensor_count, 3);

    const gobot::PhysicsSensorSnapshot& imu_snapshot = snapshot.robots[0].sensors[0];
    EXPECT_EQ(imu_snapshot.name, "imu");
    EXPECT_EQ(imu_snapshot.link_name, "base");
    EXPECT_EQ(imu_snapshot.type, gobot::PhysicsSensorType::IMU);
    ASSERT_EQ(imu_snapshot.channel_names.size(), 13);
    EXPECT_EQ(imu_snapshot.channel_names[7], "linear_velocity_x");
    EXPECT_EQ(imu_snapshot.channel_names[8], "linear_velocity_y");
    EXPECT_EQ(imu_snapshot.channel_names[9], "linear_velocity_z");
    EXPECT_TRUE(imu_snapshot.global_transform.translation().isApprox(
            gobot::Vector3(1.1, 2.2, 3.3), CMP_EPSILON));

    const gobot::PhysicsSensorSnapshot& angular_momentum_snapshot = snapshot.robots[0].sensors[1];
    EXPECT_EQ(angular_momentum_snapshot.name, "root_angmom");
    EXPECT_EQ(angular_momentum_snapshot.type, gobot::PhysicsSensorType::AngularMomentum);
    ASSERT_EQ(angular_momentum_snapshot.channel_names.size(), 3);
    EXPECT_EQ(angular_momentum_snapshot.channel_names[0], "angular_momentum_x");

    const gobot::PhysicsSensorSnapshot& contact_snapshot = snapshot.robots[0].sensors[2];
    EXPECT_EQ(contact_snapshot.name, "foot_contact");
    EXPECT_EQ(contact_snapshot.type, gobot::PhysicsSensorType::Contact);
    EXPECT_NEAR(contact_snapshot.radius, 0.05, 1.0e-6);
    EXPECT_NEAR(contact_snapshot.min_threshold, 0.1, 1.0e-6);
    EXPECT_NEAR(contact_snapshot.max_threshold, 100.0, 1.0e-6);
    ASSERT_EQ(contact_snapshot.channel_names.size(), 1);
    EXPECT_EQ(contact_snapshot.channel_names[0], "contact_strength");

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    ASSERT_EQ(state.robots[0].sensors.size(), 3);
    EXPECT_EQ(state.total_sensor_count, 3);
    ASSERT_EQ(state.robots[0].sensors[0].values.size(), 13);
    EXPECT_DOUBLE_EQ(state.robots[0].sensors[0].values[0], 1.0);
    ASSERT_EQ(state.robots[0].sensors[1].values.size(), 3);
    ASSERT_EQ(state.robots[0].sensors[2].values.size(), 1);
    EXPECT_DOUBLE_EQ(state.robots[0].sensors[2].values[0], 0.0);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, captures_terrain_nodes_in_snapshot) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->SetPosition({1.0, 2.0, 0.0});
    terrain->SetFriction({1.2, 0.01, 0.0002});
    terrain->AddBox({0.0, 0.0, -0.5}, {4.0, 3.0, 1.0});

    gobot::TerrainHeightField heightfield;
    heightfield.center = {0.5, 0.0, 0.0};
    heightfield.size = {2.0, 2.0};
    heightfield.rows = 2;
    heightfield.cols = 2;
    heightfield.heights = {0.0, 0.1, 0.2, 0.3};
    heightfield.normalized_elevation = {0.0, 0.33, 0.66, 1.0};
    heightfield.base_thickness = 0.15;
    heightfield.z_offset = -0.1;
    terrain->AddHeightField(heightfield);
    gobot::TerrainMeshPatch mesh_patch;
    mesh_patch.center = {0.0, -0.5, 0.0};
    mesh_patch.vertices = {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}};
    mesh_patch.indices = {0, 1, 2};
    mesh_patch.color = {0.5f, 0.6f, 0.7f, 1.0f};
    terrain->AddMeshPatch(mesh_patch);
    terrain->SetSpawnOrigins({{1.0, 2.0, 0.3}});
    root->AddChild(terrain);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(root));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.terrains.size(), 1);
    EXPECT_EQ(snapshot.total_terrain_count, 1);
    EXPECT_EQ(snapshot.total_collision_shape_count, 3);
    EXPECT_TRUE(snapshot.terrains[0].friction.isApprox(gobot::Vector3(1.2, 0.01, 0.0002), CMP_EPSILON));
    ASSERT_EQ(snapshot.terrains[0].boxes.size(), 1);
    ASSERT_EQ(snapshot.terrains[0].heightfields.size(), 1);
    ASSERT_EQ(snapshot.terrains[0].mesh_patches.size(), 1);
    EXPECT_TRUE(snapshot.terrains[0].boxes[0].global_transform.translation().isApprox(
            gobot::Vector3(1.0, 2.0, -0.5), CMP_EPSILON));
    EXPECT_TRUE(snapshot.terrains[0].heightfields[0].global_transform.translation().isApprox(
            gobot::Vector3(1.5, 2.0, 0.0), CMP_EPSILON));
    EXPECT_EQ(snapshot.terrains[0].heightfields[0].normalized_elevation.size(), 4);
    EXPECT_FLOAT_EQ(snapshot.terrains[0].heightfields[0].z_offset, -0.1);
    EXPECT_TRUE(snapshot.terrains[0].mesh_patches[0].global_transform.translation().isApprox(
            gobot::Vector3(1.0, 1.5, 0.0), CMP_EPSILON));
    ASSERT_EQ(snapshot.terrains[0].spawn_origins.size(), 1);
    EXPECT_TRUE(snapshot.terrains[0].spawn_origins[0].isApprox(gobot::Vector3(1.0, 2.0, 0.3), CMP_EPSILON));

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, preserves_virtual_root_link_role_in_snapshot_and_state) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* root_link = gobot::Object::New<gobot::Link3D>();
    root_link->SetName("world");
    root_link->SetRole(gobot::LinkRole::VirtualRoot);

    robot->AddChild(root_link);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1);
    ASSERT_EQ(snapshot.robots[0].links.size(), 1);
    EXPECT_EQ(snapshot.robots[0].links[0].role, gobot::PhysicsLinkRole::VirtualRoot);

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    ASSERT_EQ(state.robots[0].links.size(), 1);
    EXPECT_EQ(state.robots[0].links[0].role, gobot::PhysicsLinkRole::VirtualRoot);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, infers_implicit_virtual_root_link_from_structure) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* root_link = gobot::Object::New<gobot::Link3D>();
    root_link->SetName("world");

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("world_to_base");
    joint->SetJointType(gobot::JointType::Fixed);
    joint->SetParentLink("world");
    joint->SetChildLink("base_link");

    auto* base_link = gobot::Object::New<gobot::Link3D>();
    base_link->SetName("base_link");
    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("base_visual");
    visual->SetMesh(gobot::MakeRef<gobot::Mesh>());

    robot->AddChild(root_link);
    root_link->AddChild(joint);
    joint->AddChild(base_link);
    base_link->AddChild(visual);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1);
    ASSERT_EQ(snapshot.robots[0].links.size(), 2);
    EXPECT_EQ(snapshot.robots[0].links[0].name, "world");
    EXPECT_EQ(snapshot.robots[0].links[0].role, gobot::PhysicsLinkRole::VirtualRoot);
    EXPECT_EQ(snapshot.robots[0].links[1].name, "base_link");
    EXPECT_EQ(snapshot.robots[0].links[1].role, gobot::PhysicsLinkRole::Physical);

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    ASSERT_EQ(state.robots[0].links.size(), 2);
    EXPECT_EQ(state.robots[0].links[0].role, gobot::PhysicsLinkRole::VirtualRoot);
    EXPECT_EQ(state.robots[0].links[1].role, gobot::PhysicsLinkRole::Physical);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, keeps_root_link_with_own_visual_physical) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* root_link = gobot::Object::New<gobot::Link3D>();
    root_link->SetName("base_link");
    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("base_visual");
    visual->SetMesh(gobot::MakeRef<gobot::Mesh>());

    robot->AddChild(root_link);
    root_link->AddChild(visual);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1);
    ASSERT_EQ(snapshot.robots[0].links.size(), 1);
    EXPECT_EQ(snapshot.robots[0].links[0].role, gobot::PhysicsLinkRole::Physical);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, stores_joint_control_targets_in_scene_state) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint1");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);
    joint->SetJointPosition(0.25);
    robot->AddChild(joint);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot));

    ASSERT_TRUE(world->SetJointControl("robot", "joint1", gobot::PhysicsJointControlMode::Position, 0.5));
    const gobot::PhysicsJointState& position_state = world->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(position_state.control_mode, gobot::PhysicsJointControlMode::Position);
    EXPECT_DOUBLE_EQ(position_state.target_position, 0.5);

    ASSERT_TRUE(world->SetJointControl("robot", "joint1", gobot::PhysicsJointControlMode::Velocity, 1.25));
    const gobot::PhysicsJointState& velocity_state = world->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(velocity_state.control_mode, gobot::PhysicsJointControlMode::Velocity);
    EXPECT_DOUBLE_EQ(velocity_state.target_velocity, 1.25);

    ASSERT_TRUE(world->SetJointControl("robot", "joint1", gobot::PhysicsJointControlMode::Effort, 2.5));
    const gobot::PhysicsJointState& effort_state = world->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(effort_state.control_mode, gobot::PhysicsJointControlMode::Effort);
    EXPECT_DOUBLE_EQ(effort_state.target_effort, 2.5);

    ASSERT_TRUE(world->SetJointControl("robot", "joint1", gobot::PhysicsJointControlMode::Passive, 0.0));
    const gobot::PhysicsJointState& passive_state = world->GetSceneState().robots[0].joints[0];
    EXPECT_EQ(passive_state.control_mode, gobot::PhysicsJointControlMode::Passive);
    EXPECT_DOUBLE_EQ(passive_state.target_position, passive_state.position);
    EXPECT_DOUBLE_EQ(passive_state.target_velocity, 0.0);
    EXPECT_DOUBLE_EQ(passive_state.target_effort, 0.0);

    EXPECT_FALSE(world->SetJointControl("robot", "missing", gobot::PhysicsJointControlMode::Position, 0.0));
    EXPECT_FALSE(world->GetLastError().empty());

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

TEST(TestPhysicsServer, mujoco_external_robot_source_keeps_existing_freejoint) {
#ifdef GOBOT_HAS_MUJOCO
    const std::filesystem::path xml_path =
            std::filesystem::temp_directory_path() / "gobot_mujoco_existing_freejoint.xml";
    {
        std::ofstream file(xml_path);
        ASSERT_TRUE(file.is_open());
        file << R"(<mujoco model="free_base_bot">
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <freejoint name="floating_base_joint"/>
      <geom name="base_collision" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>)";
    }

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("free_base_bot");
    robot->SetSourcePath(xml_path.string());

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    robot->AddChild(floating_joint);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    floating_joint->AddChild(base);

    gobot::PhysicsServer physics_server(gobot::PhysicsBackendType::MuJoCoCpu);
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot)) << world->GetLastError();
    ASSERT_EQ(world->GetSceneState().robots.size(), 1);
    ASSERT_EQ(world->GetSceneState().robots[0].joints.size(), 1);
    EXPECT_EQ(world->GetSceneState().robots[0].joints[0].joint_name, "floating_base_joint");

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_authored_sensor_nodes_produce_runtime_values) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("sensor_bot");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(1.0);
    base->SetCenterOfMass({0.0, 0.0, 0.0});
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});

    auto* imu = gobot::Object::New<gobot::IMUSensor3D>();
    imu->SetName("imu");
    imu->SetPosition({0.0, 0.0, 0.0});
    base->AddChild(imu);

    auto* angular_momentum = gobot::Object::New<gobot::AngularMomentumSensor3D>();
    angular_momentum->SetName("root_angmom");
    base->AddChild(angular_momentum);

    auto* contact = gobot::Object::New<gobot::ContactSensor3D>();
    contact->SetName("contact");
    contact->SetRadius(0.05);
    contact->SetPosition({0.0, 0.0, 0.0});
    base->AddChild(contact);
    robot->AddChild(base);

    gobot::PhysicsServer physics_server(gobot::PhysicsBackendType::MuJoCoCpu);
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(robot)) << world->GetLastError();
    world->Step(0.002);

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    ASSERT_EQ(state.robots[0].sensors.size(), 3);
    EXPECT_EQ(state.total_sensor_count, 3);

    const gobot::PhysicsSensorState& imu_state = state.robots[0].sensors[0];
    EXPECT_EQ(imu_state.type, gobot::PhysicsSensorType::IMU);
    ASSERT_EQ(imu_state.values.size(), 13);
    for (gobot::RealType value : imu_state.values) {
        EXPECT_TRUE(std::isfinite(value));
    }
    EXPECT_GT(imu_state.timestamp, 0.0);

    const gobot::PhysicsSensorState& angular_momentum_state = state.robots[0].sensors[1];
    EXPECT_EQ(angular_momentum_state.type, gobot::PhysicsSensorType::AngularMomentum);
    ASSERT_EQ(angular_momentum_state.values.size(), 3);
    for (gobot::RealType value : angular_momentum_state.values) {
        EXPECT_TRUE(std::isfinite(value));
    }
    EXPECT_GT(angular_momentum_state.timestamp, 0.0);

    const gobot::PhysicsSensorState& contact_state = state.robots[0].sensors[2];
    EXPECT_EQ(contact_state.type, gobot::PhysicsSensorType::Contact);
    ASSERT_EQ(contact_state.values.size(), 1);
    EXPECT_TRUE(std::isfinite(contact_state.values[0]));
    EXPECT_GT(contact_state.timestamp, 0.0);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_authored_terrain_compiles) {
#ifdef GOBOT_HAS_MUJOCO
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->AddBox({0.0, 0.0, -0.5}, {4.0, 4.0, 1.0});

    gobot::TerrainHeightField heightfield;
    heightfield.center = {3.0, 0.0, 0.0};
    heightfield.size = {2.0, 2.0};
    heightfield.rows = 3;
    heightfield.cols = 3;
    heightfield.heights = {0.0, 0.1, 0.0, 0.2, 0.3, 0.2, 0.0, 0.1, 0.0};
    terrain->AddHeightField(heightfield);
    root->AddChild(terrain);

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("terrain_bot");
    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(1.0);
    base->SetCenterOfMass({0.0, 0.0, 0.0});
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});
    robot->AddChild(base);
    root->AddChild(robot);

    gobot::PhysicsServer physics_server(gobot::PhysicsBackendType::MuJoCoCpu);
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world->BuildFromScene(root)) << world->GetLastError();
    world->Step(0.002);
    EXPECT_EQ(world->GetSceneSnapshot().total_terrain_count, 1);
    EXPECT_EQ(world->GetSceneSnapshot().terrains[0].heightfields.size(), 1);

    gobot::Object::Delete(root);
#endif
}
