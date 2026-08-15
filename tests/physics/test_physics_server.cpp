#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <string_view>
#include <utility>

#include <gobot/physics/physics_server.hpp>
#include <gobot/physics/physics_scene_compiler.hpp>
#include <gobot/physics/backends/mujoco_scene_compiler.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/convex_mesh_shape_3d.hpp>
#include <gobot/scene/resources/mesh.hpp>
#include <gobot/scene/resources/physics_material_3d.hpp>
#include <gobot/scene/resources/sensor_noise_model.hpp>
#include <gobot/scene/rigid_body_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/sensor_3d.hpp>
#include <gobot/scene/terrain_3d.hpp>

namespace {

bool BuildWorldFromScene(const gobot::Ref<gobot::PhysicsWorld>& world, const gobot::Node* scene_root) {
    gobot::CompiledPhysicsScene compiled_scene;
    std::string error;
    if (!gobot::PhysicsSceneCompiler::Compile(scene_root, &compiled_scene, &error)) {
        return false;
    }
    return world->Build(std::move(compiled_scene.snapshot));
}

gobot::Ref<gobot::ArrayMesh> CreateTetrahedronMesh() {
    auto mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetSurface(
            {
                    {0.0, 0.0, 0.0},
                    {1.0, 0.0, 0.0},
                    {0.0, 1.0, 0.0},
                    {0.0, 0.0, 1.0},
            },
            {
                    0, 2, 1,
                    0, 1, 3,
                    1, 2, 3,
                    2, 0, 3,
            });
    return mesh;
}

std::size_t CountOccurrences(std::string_view value, std::string_view needle) {
    std::size_t count = 0;
    for (std::size_t offset = 0;
         (offset = value.find(needle, offset)) != std::string_view::npos;
         offset += needle.size()) {
        ++count;
    }
    return count;
}

} // namespace

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

    EXPECT_EQ(physics_server.GetBackendInfos().size(), 2);
}

TEST(TestPhysicsServer, creates_world_for_selected_backend) {
    gobot::PhysicsServer physics_server;

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 1.0 / 120.0;
    settings.gravity = {0.0, 0.0, -1.0};
    settings.default_joint_gains.position_stiffness = 25.0;
    settings.default_joint_gains.velocity_damping = 5.0;
    settings.default_joint_gains.integral_gain = 1.5;
    settings.default_joint_gains.integral_limit = 0.25;

    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(
            gobot::PhysicsBackendType::Null, settings);
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
    joint->SetAffineActuatorEnabled(true);
    joint->SetAffineActuatorControlGain(18.0);
    joint->SetAffineActuatorForceOffset(0.25);
    joint->SetAffineActuatorPositionGain(-18.0);
    joint->SetAffineActuatorVelocityGain(0.75);
    joint->SetAffineActuatorInheritRange(0.8);

    robot->AddChild(base_link);
    base_link->AddChild(collision_shape);
    robot->AddChild(joint);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1);
    EXPECT_EQ(snapshot.robots[0].name, "robot");
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
    EXPECT_TRUE(snapshot.robots[0].joints[0].affine_actuator_enabled);
    EXPECT_DOUBLE_EQ(snapshot.robots[0].joints[0].affine_actuator_control_gain, 18.0);
    EXPECT_DOUBLE_EQ(snapshot.robots[0].joints[0].affine_actuator_force_offset, 0.25);
    EXPECT_DOUBLE_EQ(snapshot.robots[0].joints[0].affine_actuator_position_gain, -18.0);
    EXPECT_DOUBLE_EQ(snapshot.robots[0].joints[0].affine_actuator_velocity_gain, 0.75);
    EXPECT_NEAR(snapshot.robots[0].joints[0].affine_actuator_inherit_range, 0.8, 1.0e-6);
    EXPECT_EQ(snapshot.total_link_count, 1);
    EXPECT_EQ(snapshot.total_joint_count, 1);
    EXPECT_EQ(snapshot.total_collision_shape_count, 1);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, builds_backend_neutral_indexed_topology_from_snapshot) {
    gobot::PhysicsSceneSnapshot snapshot;

    gobot::PhysicsRobotSnapshot first_robot;
    first_robot.name = "first_robot";
    gobot::PhysicsLinkSnapshot base_link;
    base_link.name = "base";
    gobot::PhysicsLinkSnapshot tip_link;
    tip_link.name = "tip";
    tip_link.role = gobot::PhysicsLinkRole::VirtualRoot;
    first_robot.links = {base_link, tip_link};

    gobot::PhysicsJointSnapshot hinge_joint;
    hinge_joint.name = "hinge";
    hinge_joint.parent_link = "base";
    hinge_joint.child_link = "tip";
    hinge_joint.joint_type = static_cast<int>(gobot::JointType::Revolute);
    gobot::PhysicsJointSnapshot dangling_joint;
    dangling_joint.name = "dangling";
    dangling_joint.parent_link = "tip";
    dangling_joint.child_link = "missing_link";
    dangling_joint.joint_type = static_cast<int>(gobot::JointType::Prismatic);
    first_robot.joints = {hinge_joint, dangling_joint};

    gobot::PhysicsRobotSnapshot second_robot;
    second_robot.name = "second_robot";
    gobot::PhysicsLinkSnapshot second_base;
    second_base.name = "root";
    second_robot.links = {second_base};

    snapshot.robots = {first_robot, second_robot};
    // The topology is derived from authored arrays instead of trusting cached
    // snapshot counters, which may come from an older serializer.
    snapshot.total_link_count = 99;
    snapshot.total_joint_count = 99;

    gobot::PhysicsServer physics_server;
    const gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(std::move(snapshot)));

    const gobot::PhysicsSceneTopology& topology = world->GetSceneTopology();
    ASSERT_EQ(topology.robots.size(), 2);
    EXPECT_EQ(topology.total_link_count, 3);
    EXPECT_EQ(topology.total_joint_count, 2);

    const gobot::PhysicsRobotTopology& first = topology.robots[0];
    EXPECT_EQ(first.index, 0);
    EXPECT_EQ(first.name, "first_robot");
    ASSERT_EQ(first.links.size(), 2);
    ASSERT_EQ(first.joints.size(), 2);
    EXPECT_EQ(first.links[0].index, 0);
    EXPECT_EQ(first.links[0].scene_index, 0);
    EXPECT_EQ(first.links[1].index, 1);
    EXPECT_EQ(first.links[1].scene_index, 1);
    EXPECT_EQ(first.links[1].role, gobot::PhysicsLinkRole::VirtualRoot);
    EXPECT_EQ(first.joints[0].index, 0);
    EXPECT_EQ(first.joints[0].scene_index, 0);
    EXPECT_EQ(first.joints[0].parent_link_index, 0);
    EXPECT_EQ(first.joints[0].child_link_index, 1);
    EXPECT_EQ(first.joints[1].parent_link_index, 1);
    EXPECT_EQ(first.joints[1].child_link_index, gobot::kInvalidPhysicsTopologyIndex);
    EXPECT_EQ(first.links[0].outgoing_joint_indices, std::vector<std::size_t>({0}));
    EXPECT_TRUE(first.links[0].incoming_joint_indices.empty());
    EXPECT_EQ(first.links[1].incoming_joint_indices, std::vector<std::size_t>({0}));
    EXPECT_EQ(first.links[1].outgoing_joint_indices, std::vector<std::size_t>({1}));

    const gobot::PhysicsRobotTopology& second = topology.robots[1];
    EXPECT_EQ(second.index, 1);
    ASSERT_EQ(second.links.size(), 1);
    EXPECT_EQ(second.links[0].scene_index, 2);
}

TEST(TestPhysicsServer, runtime_state_changes_do_not_rebuild_scene_topology) {
    gobot::PhysicsSceneSnapshot snapshot;
    gobot::PhysicsRobotSnapshot robot;
    robot.name = "robot";
    gobot::PhysicsLinkSnapshot link;
    link.name = "base";
    robot.links.push_back(link);
    gobot::PhysicsJointSnapshot joint;
    joint.name = "joint";
    joint.parent_link = "base";
    joint.joint_position = 0.25;
    robot.joints.push_back(joint);
    snapshot.robots.push_back(robot);

    gobot::PhysicsServer physics_server;
    const gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(std::move(snapshot)));

    const gobot::PhysicsSceneTopology* topology = &world->GetSceneTopology();
    ASSERT_EQ(topology->robots.size(), 1);
    ASSERT_EQ(topology->robots[0].joints.size(), 1);
    const gobot::PhysicsRobotTopology* robot_topology = &topology->robots[0];
    const gobot::PhysicsJointTopology* joint_topology = &robot_topology->joints[0];
    EXPECT_TRUE(world->ResetJointState("robot", "joint", 0.75, 0.5));
    world->Step(0.01);
    world->Reset();

    EXPECT_EQ(&world->GetSceneTopology(), topology);
    EXPECT_EQ(&topology->robots[0], robot_topology);
    EXPECT_EQ(&topology->robots[0].joints[0], joint_topology);
    EXPECT_EQ(topology->robots[0].joints[0].name, "joint");
    EXPECT_EQ(topology->robots[0].joints[0].parent_link_index, 0);
}

TEST(TestPhysicsServer, mujoco_compiles_primary_and_affine_joint_actuators) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("affine_robot");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(2.0);
    base->SetInertiaDiagonal({0.1, 0.1, 0.1});
    robot->AddChild(base);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("hinge");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetAxis(gobot::Vector3::UnitZ());
    joint->SetLowerLimit(-1.0);
    joint->SetUpperLimit(1.0);
    joint->SetDriveMode(gobot::JointDriveMode::Position);
    joint->SetDriveStiffness(30.0);
    joint->SetDriveDamping(2.0);
    joint->SetAffineActuatorEnabled(true);
    joint->SetAffineActuatorControlGain(500.0);
    joint->SetAffineActuatorPositionGain(-500.0);
    joint->SetAffineActuatorVelocityGain(1.0);
    joint->SetAffineActuatorInheritRange(1.0);
    base->AddChild(joint);

    auto* tip = gobot::Object::New<gobot::Link3D>();
    tip->SetName("tip");
    tip->SetMass(1.0);
    tip->SetCenterOfMass({0.2, 0.0, 0.0});
    tip->SetInertiaDiagonal({0.01, 0.05, 0.05});
    joint->AddChild(tip);

    gobot::PhysicsServer physics_server;
    const gobot::Ref<gobot::PhysicsWorld> world =
            physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, robot)) << world->GetLastError();

    const gobot::PhysicsSceneArtifact* artifact = world->GetSceneArtifact();
    ASSERT_NE(artifact, nullptr);
    EXPECT_EQ(artifact->schema_version, gobot::MuJoCoSceneCompiler::kArtifactSchemaVersion);
    EXPECT_EQ(artifact->producer, "mujoco");
    EXPECT_EQ(artifact->format, "mjcf");
    EXPECT_FALSE(artifact->producer_version.empty());
    EXPECT_EQ(artifact->nu, 2);
    ASSERT_EQ(artifact->robots.size(), 1);
    EXPECT_EQ(artifact->robots[0].name, "affine_robot");
    EXPECT_EQ(artifact->robots[0].runtime_prefix, "affine_robot_");
    EXPECT_EQ(artifact->robots[0].control_indices,
              std::vector<std::int32_t>({0, 1}));
    EXPECT_NE(std::find(artifact->robots[0].body_names.begin(),
                        artifact->robots[0].body_names.end(),
                        "affine_robot_tip"),
              artifact->robots[0].body_names.end());
    EXPECT_NE(std::find(artifact->robots[0].joint_names.begin(),
                        artifact->robots[0].joint_names.end(),
                        "affine_robot_hinge"),
              artifact->robots[0].joint_names.end());
    ASSERT_EQ(artifact->controls.size(), 2);
    EXPECT_EQ(artifact->controls[0].index, 0);
    EXPECT_EQ(artifact->controls[0].name, "affine_robot_hinge_position");
    EXPECT_EQ(artifact->controls[0].joint, "affine_robot_hinge");
    EXPECT_EQ(artifact->controls[0].mode, "position");
    EXPECT_EQ(artifact->controls[0].robot, "affine_robot");
    EXPECT_EQ(artifact->controls[1].index, 1);
    EXPECT_EQ(artifact->controls[1].name, "affine_robot_hinge_affine");
    EXPECT_EQ(artifact->controls[1].joint, "affine_robot_hinge");
    EXPECT_EQ(artifact->controls[1].mode, "direct");
    EXPECT_EQ(artifact->controls[1].robot, "affine_robot");
    EXPECT_NE(artifact->content.find("name=\"affine_robot_hinge_position\""),
              std::string::npos);
    EXPECT_NE(artifact->content.find("name=\"affine_robot_hinge_affine\""),
              std::string::npos);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, scene_compiler_flattens_convex_mesh_surfaces) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("mesh_robot");
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    robot->AddChild(link);

    gobot::MeshSurfaceData indexed_surface;
    indexed_surface.vertices = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {0.0, 1.0, 0.0},
    };
    indexed_surface.indices = {0, 1, 2, 0, 2, 3};
    gobot::MeshSurfaceData unindexed_surface;
    unindexed_surface.vertices = {
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0},
            {0.0, 0.0, 2.0},
            {1.0, 0.0, 2.0},
            {0.0, 1.0, 2.0},
    };
    auto mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetSurfaces({std::move(indexed_surface), std::move(unindexed_surface)});
    auto convex_shape = gobot::MakeRef<gobot::ConvexMeshShape3D>();
    convex_shape->SetMesh(mesh);

    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName("convex_collision");
    collision->SetShape(convex_shape);
    link->AddChild(collision);

    gobot::CompiledPhysicsScene compiled;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(robot, &compiled));
    ASSERT_EQ(compiled.snapshot.robots.size(), 1);
    ASSERT_EQ(compiled.snapshot.robots[0].links.size(), 1);
    ASSERT_EQ(compiled.snapshot.robots[0].links[0].collision_shapes.size(), 1);
    const gobot::PhysicsShapeSnapshot& snapshot =
            compiled.snapshot.robots[0].links[0].collision_shapes[0];
    EXPECT_EQ(snapshot.type, gobot::PhysicsShapeType::Mesh);
    ASSERT_EQ(snapshot.vertices.size(), 10);
    EXPECT_EQ(snapshot.indices,
              (std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3, 4, 5, 6, 7, 8, 9}));

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, scene_compiler_keeps_import_provenance_out_of_runtime_snapshot) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    link->SetMass(1.5);
    robot->AddChild(link);

    gobot::CompiledPhysicsScene first;
    robot->SetSourcePath("res://first.xml");
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(robot, &first));

    gobot::CompiledPhysicsScene second;
    robot->SetSourcePath("res://second.xml");
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(robot, &second));

    ASSERT_EQ(first.snapshot.robots.size(), 1);
    ASSERT_EQ(second.snapshot.robots.size(), 1);
    EXPECT_EQ(first.snapshot.robots[0].name, second.snapshot.robots[0].name);
    ASSERT_EQ(first.snapshot.robots[0].links.size(), 1);
    ASSERT_EQ(second.snapshot.robots[0].links.size(), 1);
    EXPECT_EQ(first.snapshot.robots[0].links[0].name, second.snapshot.robots[0].links[0].name);
    EXPECT_DOUBLE_EQ(first.snapshot.robots[0].links[0].mass,
                     second.snapshot.robots[0].links[0].mass);
    EXPECT_EQ(first.bindings.scene_root_id, robot->GetInstanceId());
    ASSERT_EQ(first.bindings.robots.size(), 1);
    EXPECT_EQ(first.bindings.robots[0].robot_id, robot->GetInstanceId());
    ASSERT_EQ(first.bindings.robots[0].link_ids.size(), 1);
    EXPECT_EQ(first.bindings.robots[0].link_ids[0], link->GetInstanceId());

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, scene_compiler_keeps_standalone_rigid_body_as_one_free_link) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("world");
    auto* body = gobot::Object::New<gobot::RigidBody3D>();
    body->SetName("fixture");
    body->SetMass(0.5);
    body->SetInertiaDiagonal({0.1, 0.1, 0.1});
    root->AddChild(body);
    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName("fixture_collision");
    collision->SetShape(gobot::MakeRef<gobot::BoxShape3D>());
    body->AddChild(collision);

    gobot::CompiledPhysicsScene compiled;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(root, &compiled));
    ASSERT_EQ(compiled.snapshot.robots.size(), 1u);
    const gobot::PhysicsRobotSnapshot& rigid_system = compiled.snapshot.robots[0];
    EXPECT_TRUE(rigid_system.standalone_rigid_body);
    EXPECT_EQ(rigid_system.name, "fixture");
    ASSERT_EQ(rigid_system.links.size(), 1u);
    EXPECT_TRUE(rigid_system.joints.empty());
    EXPECT_EQ(rigid_system.scene_path, rigid_system.links[0].scene_path);
    EXPECT_EQ(rigid_system.stable_id, rigid_system.links[0].stable_id);
    EXPECT_EQ(rigid_system.links[0].name, "fixture");
    ASSERT_EQ(compiled.bindings.robots.size(), 1u);
    EXPECT_EQ(compiled.bindings.robots[0].robot_id, body->GetInstanceId());
    ASSERT_EQ(compiled.bindings.robots[0].link_ids.size(), 1u);
    EXPECT_EQ(compiled.bindings.robots[0].link_ids[0], body->GetInstanceId());

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, stable_ids_are_derived_from_canonical_paths) {
    auto* root = gobot::Object::New<gobot::Node>();
    root->SetName("world");
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    root->AddChild(robot);
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    robot->AddChild(link);
    auto* sensor = gobot::Object::New<gobot::IMUSensor3D>();
    sensor->SetName("imu");
    link->AddChild(sensor);

    gobot::CompiledPhysicsScene first;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(root, &first));
    const gobot::PhysicsSensorSnapshot& first_sensor =
            first.snapshot.robots[0].sensors[0];
    EXPECT_EQ(first_sensor.scene_path, "/world/robot/base/imu");
    EXPECT_NE(first_sensor.stable_id, 0u);

    auto* unrelated = gobot::Object::New<gobot::Node>();
    unrelated->SetName("aaa_unrelated");
    root->AddChild(unrelated);
    gobot::CompiledPhysicsScene second;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(root, &second));
    const gobot::PhysicsSensorSnapshot& second_sensor =
            second.snapshot.robots[0].sensors[0];
    EXPECT_EQ(second_sensor.scene_path, first_sensor.scene_path);
    EXPECT_EQ(second_sensor.stable_id, first_sensor.stable_id);

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, scene_compiler_assigns_unique_identities_to_duplicate_authored_names) {
    auto* root = gobot::Object::New<gobot::Node>();
    root->SetName("world");
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    root->AddChild(robot);
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    robot->AddChild(link);
    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName("shared_name");
    collision->SetShape(gobot::MakeRef<gobot::BoxShape3D>());
    link->AddChild(collision);
    auto* sensor = gobot::Object::New<gobot::IMUSensor3D>();
    sensor->SetName("shared_name");
    link->AddChild(sensor);

    gobot::CompiledPhysicsScene compiled;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(root, &compiled));
    ASSERT_EQ(compiled.snapshot.robots.size(), 1u);
    ASSERT_EQ(compiled.snapshot.robots[0].links.size(), 1u);
    ASSERT_EQ(compiled.snapshot.robots[0].links[0].collision_shapes.size(), 1u);
    ASSERT_EQ(compiled.snapshot.robots[0].sensors.size(), 1u);
    const gobot::PhysicsShapeSnapshot& shape =
            compiled.snapshot.robots[0].links[0].collision_shapes[0];
    const gobot::PhysicsSensorSnapshot& sensor_snapshot =
            compiled.snapshot.robots[0].sensors[0];
    EXPECT_NE(shape.scene_path, sensor_snapshot.scene_path);
    EXPECT_NE(shape.stable_id, sensor_snapshot.stable_id);

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, scene_compiler_validates_loose_sensor_noise) {
    auto* root = gobot::Object::New<gobot::Node>();
    root->SetName("world");
    auto* sensor = gobot::Object::New<gobot::IMUSensor3D>();
    sensor->SetName("imu");
    auto noise = gobot::MakeRef<gobot::SensorNoiseModel>();
    noise->SetWhiteNoiseStddev(-0.01);
    sensor->SetNoiseModel(noise);
    root->AddChild(sensor);

    gobot::CompiledPhysicsScene compiled;
    std::string error;
    EXPECT_FALSE(gobot::PhysicsSceneCompiler::Compile(root, &compiled, &error));
    EXPECT_NE(error.find("SensorNoiseModel"), std::string::npos);

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, scene_compiler_captures_nested_link_shapes_and_sensors) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    auto* mount = gobot::Object::New<gobot::Node3D>();
    mount->SetName("sensor_mount");
    mount->SetPosition({0.1, 0.2, 0.3});

    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName("nested_collision");
    collision->SetPosition({0.0, 0.0, -0.1});
    auto shape = gobot::MakeRef<gobot::BoxShape3D>();
    shape->SetSize({0.2, 0.3, 0.4});
    collision->SetShape(shape);

    auto* imu = gobot::Object::New<gobot::IMUSensor3D>();
    imu->SetName("nested_imu");
    imu->SetPosition({0.0, 0.0, 0.1});

    robot->AddChild(link);
    link->AddChild(mount);
    mount->AddChild(collision);
    mount->AddChild(imu);

    gobot::CompiledPhysicsScene compiled_scene;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(robot, &compiled_scene));
    ASSERT_EQ(compiled_scene.snapshot.robots.size(), 1);
    ASSERT_EQ(compiled_scene.snapshot.robots[0].links.size(), 1);
    ASSERT_EQ(compiled_scene.snapshot.robots[0].links[0].collision_shapes.size(), 1);
    EXPECT_EQ(compiled_scene.snapshot.robots[0].links[0].collision_shapes[0].name,
              "nested_collision");
    EXPECT_TRUE(compiled_scene.snapshot.robots[0].links[0].collision_shapes[0]
                        .global_transform.translation().isApprox(
                                gobot::Vector3(0.1, 0.2, 0.2), CMP_EPSILON));
    ASSERT_EQ(compiled_scene.snapshot.robots[0].sensors.size(), 1);
    EXPECT_EQ(compiled_scene.snapshot.robots[0].sensors[0].name, "nested_imu");
    EXPECT_EQ(compiled_scene.snapshot.robots[0].sensors[0].link_name, "base");
    EXPECT_TRUE(compiled_scene.snapshot.robots[0].sensors[0]
                        .global_transform.translation().isApprox(
                                gobot::Vector3(0.1, 0.2, 0.4), CMP_EPSILON));

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, scene_compiler_rejects_duplicate_runtime_names) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");
    for (int index = 0; index < 2; ++index) {
        auto* group = gobot::Object::New<gobot::Node3D>();
        group->SetName(index == 0 ? "first_group" : "second_group");
        auto* robot = gobot::Object::New<gobot::Robot3D>();
        robot->SetName("robot");
        auto* link = gobot::Object::New<gobot::Link3D>();
        link->SetName(index == 0 ? "base" : "other_base");
        robot->AddChild(link);
        group->AddChild(robot);
        root->AddChild(group);
    }

    gobot::CompiledPhysicsScene compiled_scene;
    std::string error;
    EXPECT_FALSE(gobot::PhysicsSceneCompiler::Compile(root, &compiled_scene, &error));
    EXPECT_NE(error.find("Duplicate robot name 'robot'"), std::string::npos);
    EXPECT_TRUE(std::any_of(
            compiled_scene.diagnostics.begin(),
            compiled_scene.diagnostics.end(),
            [](const gobot::PhysicsSceneCompileDiagnostic& diagnostic) {
                return diagnostic.severity == gobot::PhysicsSceneCompileSeverity::Error;
            }));

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, scene_compiler_rejects_duplicate_named_collision_shapes_within_robot) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    for (int index = 0; index < 2; ++index) {
        auto* link = gobot::Object::New<gobot::Link3D>();
        link->SetName(index == 0 ? "base" : "tip");
        auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
        collision->SetName("shared_collision");
        collision->SetShape(gobot::MakeRef<gobot::BoxShape3D>());
        link->AddChild(collision);
        robot->AddChild(link);
    }

    gobot::CompiledPhysicsScene compiled_scene;
    std::string error;
    EXPECT_FALSE(gobot::PhysicsSceneCompiler::Compile(robot, &compiled_scene, &error));
    EXPECT_NE(error.find("Duplicate collision shape name 'shared_collision'"), std::string::npos);

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
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

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
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

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

    auto* terrain_height = gobot::Object::New<gobot::HeightScanner3D>();
    terrain_height->SetName("terrain_scan");
    terrain_height->SetPosition({0.0, 0.0, 0.4});
    terrain_height->SetSampleOffsets({{0.0, 0.0, 0.0}, {0.2, 0.0, 0.0}});
    terrain_height->SetRayDirection({0.0, 0.0, -1.0});
    terrain_height->SetMaxDistance(2.0);
    base_link->AddChild(terrain_height);
    robot->AddChild(base_link);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1);
    ASSERT_EQ(snapshot.robots[0].sensors.size(), 4);
    EXPECT_EQ(snapshot.total_sensor_count, 4);

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

    const gobot::PhysicsSensorSnapshot& terrain_height_snapshot = snapshot.robots[0].sensors[3];
    EXPECT_EQ(terrain_height_snapshot.name, "terrain_scan");
    EXPECT_EQ(terrain_height_snapshot.type, gobot::PhysicsSensorType::HeightScanner);
    ASSERT_EQ(terrain_height_snapshot.sample_offsets.size(), 2);
    EXPECT_TRUE(terrain_height_snapshot.ray_direction.isApprox(gobot::Vector3(0.0, 0.0, -1.0), CMP_EPSILON));
    EXPECT_TRUE(terrain_height_snapshot.ray_direction_world_space);
    EXPECT_NEAR(terrain_height_snapshot.max_distance, 2.0, 1.0e-6);
    ASSERT_EQ(terrain_height_snapshot.channel_names.size(), 2);
    EXPECT_EQ(terrain_height_snapshot.channel_names[0], "distance_0");
    EXPECT_EQ(terrain_height_snapshot.channel_names[1], "distance_1");
    EXPECT_TRUE(terrain_height_snapshot.local_transform.translation().isApprox(
            gobot::Vector3(0.0, 0.0, 0.4), CMP_EPSILON));

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    ASSERT_EQ(state.robots[0].sensors.size(), 4);
    EXPECT_EQ(state.total_sensor_count, 4);
    ASSERT_EQ(state.robots[0].sensors[0].values.size(), 13);
    EXPECT_DOUBLE_EQ(state.robots[0].sensors[0].values[0], 1.0);
    ASSERT_EQ(state.robots[0].sensors[1].values.size(), 3);
    ASSERT_EQ(state.robots[0].sensors[2].values.size(), 1);
    EXPECT_DOUBLE_EQ(state.robots[0].sensors[2].values[0], 0.0);
    ASSERT_EQ(state.robots[0].sensors[3].values.size(), 2);
    ASSERT_EQ(state.robots[0].sensors[3].hits.size(), 2);
    EXPECT_TRUE(state.robots[0].sensors[3].global_transform.translation().isApprox(
            gobot::Vector3(1.0, 2.0, 3.4), CMP_EPSILON));

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, height_scanner_raycast_queries_box_heightfield_and_mesh_patch) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->AddBox({0.0, 0.0, -0.1}, {1.0, 1.0, 0.2});

    gobot::TerrainHeightField heightfield;
    heightfield.center = {2.0, 0.0, 0.0};
    heightfield.size = {2.0, 2.0};
    heightfield.rows = 2;
    heightfield.cols = 2;
    heightfield.heights = {0.0, 0.2, 0.4, 0.6};
    terrain->AddHeightField(heightfield);

    gobot::TerrainMeshPatch mesh_patch;
    mesh_patch.center = {4.0, 0.0, 0.0};
    mesh_patch.vertices = {{-0.5, -0.5, 0.3}, {0.5, -0.5, 0.3}, {-0.5, 0.5, 0.3}};
    mesh_patch.indices = {0, 1, 2};
    terrain->AddMeshPatch(mesh_patch);
    root->AddChild(terrain);

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    link->SetPosition({0.0, 0.0, 1.0});
    auto* sensor = gobot::Object::New<gobot::HeightScanner3D>();
    sensor->SetName("terrain_scan");
    sensor->SetSampleOffsets({{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {4.0, -0.25, 0.0}});
    sensor->SetMaxDistance(2.0);
    link->AddChild(sensor);
    robot->AddChild(link);
    root->AddChild(robot);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(BuildWorldFromScene(world, root));
    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.terrains.size(), 1);
    ASSERT_EQ(snapshot.terrains[0].boxes.size(), 1);
    ASSERT_EQ(snapshot.terrains[0].heightfields.size(), 1);
    ASSERT_EQ(snapshot.terrains[0].mesh_patches.size(), 1);
    EXPECT_TRUE(snapshot.terrains[0].boxes[0].has_xy_bounds);
    EXPECT_TRUE(snapshot.terrains[0].heightfields[0].has_xy_bounds);
    EXPECT_TRUE(snapshot.terrains[0].mesh_patches[0].has_xy_bounds);
    EXPECT_TRUE(snapshot.terrains[0].heightfields[0].xy_min.isApprox(gobot::Vector2(1.0, -1.0), CMP_EPSILON));
    EXPECT_TRUE(snapshot.terrains[0].heightfields[0].xy_max.isApprox(gobot::Vector2(3.0, 1.0), CMP_EPSILON));

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.robots.size(), 1);
    ASSERT_EQ(state.robots[0].sensors.size(), 1);
    const gobot::PhysicsSensorState& sensor_state = state.robots[0].sensors[0];
    ASSERT_EQ(sensor_state.values.size(), 3);
    EXPECT_NEAR(sensor_state.values[0], 1.0, 1.0e-6);
    EXPECT_NEAR(sensor_state.values[1], 0.7, 1.0e-6);
    EXPECT_NEAR(sensor_state.values[2], 0.7, 1.0e-6);
    ASSERT_EQ(sensor_state.hits.size(), 3);
    EXPECT_TRUE(sensor_state.hits[0].hit);
    EXPECT_TRUE(sensor_state.hits[1].hit);
    EXPECT_TRUE(sensor_state.hits[2].hit);
    EXPECT_NEAR(sensor_state.hits[0].point.x(), 0.0, 1.0e-6);
    EXPECT_NEAR(sensor_state.hits[0].point.y(), 0.0, 1.0e-6);
    EXPECT_NEAR(sensor_state.hits[0].point.z(), 0.0, 1.0e-6);
    EXPECT_TRUE(sensor_state.hits[1].normal.z() < 0.99);
    EXPECT_TRUE(sensor_state.hits[1].normal.z() > 0.0);

    const gobot::PhysicsRaycastHit heightfield_hit = world->RaycastTerrain({
            {2.0, 0.0, 1.0},
            {0.0, 0.0, -1.0},
            2.0
    });
    EXPECT_TRUE(heightfield_hit.hit);
    EXPECT_NEAR(heightfield_hit.point.z(), 0.3, 1.0e-6);

    const gobot::PhysicsRaycastHit heightfield_bounds_miss = world->RaycastTerrain({
            {3.25, 0.0, 1.0},
            {0.0, 0.0, -1.0},
            2.0
    });
    EXPECT_FALSE(heightfield_bounds_miss.hit);

    ASSERT_TRUE(world->ResetLinkState("robot", "base", {0.0, 0.0, 1.2}));
    const gobot::PhysicsSensorState& moved_sensor_state = world->GetSceneState().robots[0].sensors[0];
    EXPECT_NEAR(moved_sensor_state.values[0], 1.2, 1.0e-6);
    EXPECT_TRUE(moved_sensor_state.global_transform.translation().isApprox(
            gobot::Vector3(0.0, 0.0, 1.2), CMP_EPSILON));

    auto* loose_sensor = gobot::Object::New<gobot::TerrainHeightSensor3D>();
    loose_sensor->SetName("preview_height");
    loose_sensor->SetPosition({0.0, 0.0, 1.0});
    loose_sensor->SetSampleOffsets({{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}});
    loose_sensor->SetMaxDistance(2.0);
    root->AddChild(loose_sensor);

    ASSERT_TRUE(BuildWorldFromScene(world, root));
    const gobot::PhysicsSceneState& preview_state = world->GetSceneState();
    ASSERT_EQ(preview_state.loose_sensors.size(), 1);
    const gobot::PhysicsSensorState& loose_sensor_state = preview_state.loose_sensors[0];
    EXPECT_EQ(loose_sensor_state.type, gobot::PhysicsSensorType::TerrainHeight);
    ASSERT_EQ(loose_sensor_state.values.size(), 1);
    EXPECT_NEAR(loose_sensor_state.values[0], 0.7, 1.0e-6);
    ASSERT_EQ(loose_sensor_state.hits.size(), 2);
    EXPECT_TRUE(loose_sensor_state.hits[0].hit);
    EXPECT_TRUE(loose_sensor_state.hits[1].hit);

    const gobot::PhysicsRaycastHit miss = world->RaycastTerrain({
            {10.0, 10.0, 1.0},
            {0.0, 0.0, -1.0},
            1.5
    });
    EXPECT_FALSE(miss.hit);
    EXPECT_NEAR(miss.distance, 1.5, 1.0e-6);

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, sensor_noise_is_deterministic_across_reset_and_checkpoint_restore) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("ground");
    terrain->AddBox({0.0, 0.0, -0.1}, {2.0, 2.0, 0.2});
    root->AddChild(terrain);

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    link->SetPosition({0.0, 0.0, 1.0});
    auto* sensor = gobot::Object::New<gobot::TerrainHeightSensor3D>();
    sensor->SetName("height");
    sensor->SetSampleOffsets({{0.0, 0.0, 0.0}});
    sensor->SetMaxDistance(2.0);
    auto noise = gobot::MakeRef<gobot::SensorNoiseModel>();
    noise->SetWhiteNoiseStddev(0.02);
    noise->SetBiasMean(0.03);
    noise->SetBiasStddev(0.01);
    noise->SetRandomWalkStddev(0.1);
    noise->SetSeedOffset(17);
    sensor->SetNoiseModel(noise);
    link->AddChild(sensor);
    robot->AddChild(link);
    root->AddChild(robot);

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.01;
    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(
            gobot::PhysicsBackendType::Null, settings);
    ASSERT_TRUE(BuildWorldFromScene(world, root));

    const gobot::PhysicsSensorState initial = world->GetSceneState().robots[0].sensors[0];
    ASSERT_EQ(initial.sample_count, 1);
    ASSERT_EQ(initial.noise_bias.size(), 1);
    ASSERT_EQ(initial.noise_random_walk.size(), 1);
    const gobot::Ref<gobot::PhysicsRuntimeCheckpoint> checkpoint = world->CaptureCheckpoint();
    ASSERT_TRUE(checkpoint.IsValid());

    ASSERT_TRUE(world->ResetLinkState("robot", "base", {0.0, 0.0, 1.25}));
    const gobot::PhysicsSensorState advanced = world->GetSceneState().robots[0].sensors[0];
    ASSERT_EQ(advanced.sample_count, 2);
    EXPECT_NE(advanced.values[0], initial.values[0]);

    ASSERT_TRUE(world->RestoreCheckpoint(checkpoint));
    const gobot::PhysicsSensorState restored = world->GetSceneState().robots[0].sensors[0];
    EXPECT_EQ(restored.sample_count, initial.sample_count);
    EXPECT_DOUBLE_EQ(restored.values[0], initial.values[0]);
    EXPECT_DOUBLE_EQ(restored.noise_bias[0], initial.noise_bias[0]);
    EXPECT_DOUBLE_EQ(restored.noise_random_walk[0], initial.noise_random_walk[0]);

    ASSERT_TRUE(world->ResetLinkState("robot", "base", {0.0, 0.0, 1.25}));
    const gobot::PhysicsSensorState replayed = world->GetSceneState().robots[0].sensors[0];
    EXPECT_DOUBLE_EQ(replayed.values[0], advanced.values[0]);
    EXPECT_DOUBLE_EQ(replayed.noise_random_walk[0], advanced.noise_random_walk[0]);

    world->Reset();
    const gobot::PhysicsSensorState reset = world->GetSceneState().robots[0].sensors[0];
    EXPECT_DOUBLE_EQ(reset.values[0], initial.values[0]);
    EXPECT_DOUBLE_EQ(reset.noise_bias[0], initial.noise_bias[0]);
    EXPECT_EQ(reset.sample_count, initial.sample_count);

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, sensor_noise_quantizes_then_clips_samples) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");
    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("ground");
    terrain->AddBox({0.0, 0.0, -0.1}, {2.0, 2.0, 0.2});
    root->AddChild(terrain);
    auto* sensor = gobot::Object::New<gobot::TerrainHeightSensor3D>();
    sensor->SetName("height");
    sensor->SetPosition({0.0, 0.0, 1.0});
    sensor->SetSampleOffsets({{0.0, 0.0, 0.0}});
    sensor->SetMaxDistance(2.0);
    auto noise = gobot::MakeRef<gobot::SensorNoiseModel>();
    noise->SetBiasMean(0.137);
    noise->SetQuantizationStep(0.05);
    noise->SetClipMin(0.0);
    noise->SetClipMax(1.12);
    sensor->SetNoiseModel(noise);
    root->AddChild(sensor);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(BuildWorldFromScene(world, root));
    const gobot::PhysicsSensorState& state = world->GetSceneState().loose_sensors[0];
    ASSERT_EQ(state.values.size(), 1);
    EXPECT_NEAR(state.values[0], 1.12, 1.0e-7);

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, checkpoint_rejects_changed_backend_neutral_runtime_contract) {
    auto* root = gobot::Object::New<gobot::Node>();
    root->SetName("world");
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    root->AddChild(robot);
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    robot->AddChild(link);
    auto* sensor = gobot::Object::New<gobot::IMUSensor3D>();
    sensor->SetName("imu");
    link->AddChild(sensor);
    auto noise = gobot::MakeRef<gobot::SensorNoiseModel>();
    noise->SetWhiteNoiseStddev(0.01);
    sensor->SetNoiseModel(noise);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world =
            physics_server.CreateWorld(gobot::PhysicsBackendType::Null);
    ASSERT_TRUE(BuildWorldFromScene(world, root)) << world->GetLastError();
    gobot::Ref<gobot::PhysicsRuntimeCheckpoint> checkpoint = world->CaptureCheckpoint();
    ASSERT_TRUE(checkpoint.IsValid());

    noise->SetWhiteNoiseStddev(0.02);
    ASSERT_TRUE(BuildWorldFromScene(world, root)) << world->GetLastError();
    EXPECT_FALSE(world->RestoreCheckpoint(checkpoint));
    EXPECT_NE(world->GetLastError().find("artifact digest"), std::string::npos);

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, terrain_height_sensor_falls_back_to_clamped_frame_height_when_all_rays_miss) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    link->SetPosition({0.0, 0.0, 0.4});
    auto* sensor = gobot::Object::New<gobot::TerrainHeightSensor3D>();
    sensor->SetName("foot_height");
    sensor->SetSampleOffsets({{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}});
    sensor->SetMaxDistance(1.0);
    sensor->SetReductionMode(gobot::RayReductionMode::Min);
    link->AddChild(sensor);
    robot->AddChild(link);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(BuildWorldFromScene(world, robot));
    const gobot::PhysicsSensorState& sensor_state =
            world->GetSceneState().robots[0].sensors[0];
    ASSERT_EQ(sensor_state.values.size(), 1);
    EXPECT_NEAR(sensor_state.values[0], 0.4, 1.0e-6);
    ASSERT_EQ(sensor_state.hits.size(), 2);
    EXPECT_FALSE(sensor_state.hits[0].hit);
    EXPECT_FALSE(sensor_state.hits[1].hit);

    gobot::Object::Delete(robot);
}

TEST(TestPhysicsServer, raycast_grid_pattern_can_follow_yaw_without_pitch_roll) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->AddBox({0.0, 0.0, -0.1}, {10.0, 10.0, 0.2});
    root->AddChild(terrain);

    auto* sensor = gobot::Object::New<gobot::RayCastSensor3D>();
    sensor->SetName("terrain_scan");
    sensor->SetPosition({0.0, 0.0, 1.0});
    sensor->SetEulerDegree({20.0, 30.0, 90.0});
    sensor->SetPatternMode(gobot::RayPatternMode::Grid);
    sensor->SetGridSize({0.2, 0.2});
    sensor->SetGridResolution(0.2);
    sensor->SetRayAlignment(gobot::RayAlignmentMode::Yaw);
    sensor->SetMaxDistance(2.0);
    root->AddChild(sensor);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld();
    ASSERT_TRUE(BuildWorldFromScene(world, root));

    const gobot::PhysicsSceneState& state = world->GetSceneState();
    ASSERT_EQ(state.loose_sensors.size(), 1);
    const gobot::PhysicsSensorState& sensor_state = state.loose_sensors[0];
    ASSERT_EQ(sensor_state.hits.size(), 4);
    for (const gobot::PhysicsSensorRaycastHit& hit : sensor_state.hits) {
        EXPECT_TRUE(hit.hit);
        EXPECT_NEAR(hit.origin.z(), 1.0, 1.0e-6);
        EXPECT_NEAR(hit.point.z(), 0.0, 1.0e-6);
    }

    gobot::Object::Delete(root);
}

TEST(TestPhysicsServer, captures_terrain_nodes_in_snapshot) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->SetPosition({1.0, 2.0, 0.0});
    auto terrain_material = gobot::MakeRef<gobot::PhysicsMaterial3D>();
    terrain_material->SetSlidingFriction(1.2);
    terrain_material->SetTorsionalFriction(0.01);
    terrain_material->SetRollingFriction(0.0002);
    terrain->SetPhysicsMaterial(terrain_material);
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
    ASSERT_TRUE(BuildWorldFromScene(world, root));

    const gobot::PhysicsSceneSnapshot& snapshot = world->GetSceneSnapshot();
    ASSERT_EQ(snapshot.terrains.size(), 1);
    EXPECT_EQ(snapshot.total_terrain_count, 1);
    EXPECT_EQ(snapshot.total_collision_shape_count, 3);
    EXPECT_NEAR(snapshot.terrains[0].material.sliding_friction, 1.2, CMP_EPSILON);
    EXPECT_NEAR(snapshot.terrains[0].material.torsional_friction, 0.01, CMP_EPSILON);
    EXPECT_NEAR(snapshot.terrains[0].material.rolling_friction, 0.0002, CMP_EPSILON);
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
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

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
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

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
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

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
    ASSERT_TRUE(BuildWorldFromScene(world, robot));

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
    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);

#ifndef GOBOT_HAS_MUJOCO
    EXPECT_FALSE(world.IsValid());
    const gobot::PhysicsBackendInfo info =
            physics_server.GetBackendInfo(gobot::PhysicsBackendType::MuJoCoCpu);
    EXPECT_FALSE(info.available);
    EXPECT_FALSE(info.status.empty());
#else
    ASSERT_TRUE(world.IsValid());
    EXPECT_EQ(world->GetBackendType(), gobot::PhysicsBackendType::MuJoCoCpu);
    EXPECT_TRUE(world->IsAvailable());
#endif
}

TEST(TestPhysicsServer, mujoco_compiles_authored_floating_joint) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("free_base_bot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    robot->AddChild(floating_joint);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    floating_joint->AddChild(base);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, robot)) << world->GetLastError();
    ASSERT_EQ(world->GetSceneState().robots.size(), 1);
    ASSERT_EQ(world->GetSceneState().robots[0].joints.size(), 1);
    EXPECT_EQ(world->GetSceneState().robots[0].joints[0].joint_name, "floating_base_joint");

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_compiles_standalone_rigid_body_with_internal_free_joint) {
#ifdef GOBOT_HAS_MUJOCO
    auto* body = gobot::Object::New<gobot::RigidBody3D>();
    body->SetName("fixture");
    body->SetMass(0.5);
    body->SetInertiaDiagonal({0.1, 0.1, 0.1});
    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName("fixture_collision");
    collision->SetShape(gobot::MakeRef<gobot::BoxShape3D>());
    body->AddChild(collision);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world =
            physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, body)) << world->GetLastError();
    const gobot::PhysicsSceneArtifact* artifact = world->GetSceneArtifact();
    ASSERT_NE(artifact, nullptr);
    EXPECT_EQ(artifact->nq, 7);
    EXPECT_EQ(artifact->nv, 6);
    EXPECT_NE(artifact->content.find("fixture_free_joint"), std::string::npos);
    ASSERT_EQ(world->GetSceneState().robots.size(), 1u);
    EXPECT_EQ(world->GetSceneState().robots[0].name, "fixture");
    EXPECT_TRUE(world->GetSceneState().robots[0].joints.empty());
    ASSERT_TRUE(world->ResetLinkState(
            "fixture", "fixture", {0.2, -0.1, 0.8},
            gobot::Quaternion::Identity(), {0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6}));
    const gobot::PhysicsLinkState& state =
            world->GetSceneState().robots[0].links[0];
    EXPECT_TRUE(state.global_transform.translation().isApprox(
            gobot::Vector3(0.2, -0.1, 0.8)));
    EXPECT_TRUE(state.linear_velocity.isApprox(gobot::Vector3(0.1, 0.2, 0.3)));
    EXPECT_TRUE(state.angular_velocity.isApprox(gobot::Vector3(0.4, 0.5, 0.6)));

    gobot::Object::Delete(body);
#endif
}

TEST(TestPhysicsServer, mujoco_exposes_backend_neutral_batch_robot_state) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("batch_robot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    robot->AddChild(floating_joint);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(1.0);
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});
    floating_joint->AddChild(base);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, robot)) << world->GetLastError();
    ASSERT_TRUE(world->ConfigureEnvironmentBatch(3)) << world->GetLastError();

    gobot::PhysicsRobotBatchStepRequest request;
    request.robot_name = "batch_robot";
    request.base_link = "base";
    request.link_names = {"base"};
    request.ticks = 2;
    request.worker_count = 2;
    gobot::PhysicsRobotBatchStepResult arrays;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    EXPECT_EQ(arrays.environment_count, 3);
    EXPECT_EQ(arrays.robot_name, "batch_robot");
    EXPECT_EQ(arrays.base_link, "base");
    ASSERT_EQ(arrays.base_position.size(), 9);
    ASSERT_EQ(arrays.base_quaternion.size(), 12);
    ASSERT_EQ(arrays.link_position.size(), 9);
    for (gobot::RealType value : arrays.base_position) {
        EXPECT_TRUE(std::isfinite(value));
    }

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_environment_batch_matches_single_ticks_for_software_pd_and_actual_dt) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("pd_parity_robot");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(5.0);
    base->SetInertiaDiagonal({1.0, 1.0, 1.0});
    robot->AddChild(base);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("hinge");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetAxis({0.0, 0.0, 1.0});
    joint->SetLowerLimit(-1.5);
    joint->SetUpperLimit(1.5);
    joint->SetEffortLimit(100.0);
    base->AddChild(joint);

    auto* tip = gobot::Object::New<gobot::Link3D>();
    tip->SetName("tip");
    tip->SetMass(1.0);
    tip->SetCenterOfMass({0.3, 0.0, 0.0});
    tip->SetInertiaDiagonal({0.01, 0.05, 0.05});
    joint->AddChild(tip);

    constexpr gobot::RealType step_delta_time = 0.002;
    constexpr std::uint64_t tick_count = 25;
    constexpr std::size_t environment_count = 2;

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.02;
    settings.gravity = gobot::Vector3::Zero();
    settings.default_joint_gains.position_stiffness = 8.0;
    settings.default_joint_gains.velocity_damping = 0.5;
    settings.default_joint_gains.integral_gain = 30.0;
    settings.default_joint_gains.integral_limit = 1.0;
    gobot::PhysicsWorldSettings actual_dt_settings = settings;
    actual_dt_settings.fixed_time_step = step_delta_time;

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> batch_world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu, settings);
    gobot::Ref<gobot::PhysicsWorld> single_world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu, settings);
    gobot::Ref<gobot::PhysicsWorld> actual_dt_reference_world =
            physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu, actual_dt_settings);
    ASSERT_TRUE(BuildWorldFromScene(batch_world, robot)) << batch_world->GetLastError();
    ASSERT_TRUE(BuildWorldFromScene(single_world, robot)) << single_world->GetLastError();
    ASSERT_TRUE(BuildWorldFromScene(actual_dt_reference_world, robot))
            << actual_dt_reference_world->GetLastError();
    ASSERT_TRUE(batch_world->ConfigureEnvironmentBatch(environment_count)) << batch_world->GetLastError();
    ASSERT_TRUE(single_world->ConfigureEnvironmentBatch(environment_count)) << single_world->GetLastError();
    ASSERT_TRUE(actual_dt_reference_world->ConfigureEnvironmentBatch(environment_count))
            << actual_dt_reference_world->GetLastError();

    const std::array<gobot::RealType, environment_count> targets{0.75, -0.4};
    for (std::size_t environment_index = 0; environment_index < environment_count; ++environment_index) {
        for (const gobot::Ref<gobot::PhysicsWorld>& world :
             {batch_world, single_world, actual_dt_reference_world}) {
            ASSERT_TRUE(world->SetEnvironmentJointControl(environment_index,
                                                          "pd_parity_robot",
                                                          "hinge",
                                                          gobot::PhysicsJointControlMode::Position,
                                                          targets[environment_index]))
                    << world->GetLastError();
        }
    }

    ASSERT_TRUE(batch_world->StepEnvironmentBatch(step_delta_time, tick_count, 2))
            << batch_world->GetLastError();
    for (std::uint64_t tick = 0; tick < tick_count; ++tick) {
        for (std::size_t environment_index = 0; environment_index < environment_count; ++environment_index) {
            ASSERT_TRUE(single_world->StepEnvironment(environment_index, step_delta_time))
                    << single_world->GetLastError();
            ASSERT_TRUE(actual_dt_reference_world->StepEnvironment(environment_index, step_delta_time))
                    << actual_dt_reference_world->GetLastError();
        }
    }

    for (std::size_t environment_index = 0; environment_index < environment_count; ++environment_index) {
        const gobot::PhysicsSceneState* batch_state = batch_world->GetEnvironmentState(environment_index);
        const gobot::PhysicsSceneState* single_state = single_world->GetEnvironmentState(environment_index);
        const gobot::PhysicsSceneState* actual_dt_state =
                actual_dt_reference_world->GetEnvironmentState(environment_index);
        ASSERT_NE(batch_state, nullptr);
        ASSERT_NE(single_state, nullptr);
        ASSERT_NE(actual_dt_state, nullptr);
        ASSERT_EQ(batch_state->robots[0].joints.size(), 1);
        const gobot::PhysicsJointState& batch_joint = batch_state->robots[0].joints[0];
        const gobot::PhysicsJointState& single_joint = single_state->robots[0].joints[0];
        const gobot::PhysicsJointState& actual_dt_joint = actual_dt_state->robots[0].joints[0];
        EXPECT_NEAR(batch_joint.position, single_joint.position, 1.0e-10);
        EXPECT_NEAR(batch_joint.velocity, single_joint.velocity, 1.0e-10);
        EXPECT_NEAR(single_joint.position, actual_dt_joint.position, 1.0e-10);
        EXPECT_NEAR(single_joint.velocity, actual_dt_joint.velocity, 1.0e-10);
        EXPECT_GT(std::abs(batch_joint.position), 1.0e-6);
    }

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_environment_batch_recomputes_spring_wrench_each_tick) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("spring_parity_robot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    robot->AddChild(floating_joint);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(1.0);
    base->SetInertiaDiagonal({0.05, 0.05, 0.05});
    floating_joint->AddChild(base);

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.01;
    settings.gravity = gobot::Vector3::Zero();
    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> batch_world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu, settings);
    gobot::Ref<gobot::PhysicsWorld> single_world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu, settings);
    ASSERT_TRUE(BuildWorldFromScene(batch_world, robot)) << batch_world->GetLastError();
    ASSERT_TRUE(BuildWorldFromScene(single_world, robot)) << single_world->GetLastError();
    ASSERT_TRUE(batch_world->ConfigureEnvironmentBatch(2)) << batch_world->GetLastError();
    ASSERT_TRUE(single_world->ConfigureEnvironmentBatch(2)) << single_world->GetLastError();

    for (const gobot::Ref<gobot::PhysicsWorld>& world : {batch_world, single_world}) {
        ASSERT_TRUE(world->SetLinkSpringForce("spring_parity_robot",
                                              "base",
                                              gobot::Vector3::Zero(),
                                              {0.5, 0.0, 0.0},
                                              {1.0, 0.0, 0.0}))
                << world->GetLastError();
    }

    constexpr gobot::RealType delta_time = 0.004;
    constexpr std::uint64_t tick_count = 40;
    ASSERT_TRUE(batch_world->StepEnvironmentBatch(delta_time, tick_count, 2))
            << batch_world->GetLastError();
    for (std::uint64_t tick = 0; tick < tick_count; ++tick) {
        for (std::size_t environment_index = 0; environment_index < 2; ++environment_index) {
            ASSERT_TRUE(single_world->StepEnvironment(environment_index, delta_time))
                    << single_world->GetLastError();
        }
    }

    for (std::size_t environment_index = 0; environment_index < 2; ++environment_index) {
        const gobot::PhysicsSceneState* batch_state = batch_world->GetEnvironmentState(environment_index);
        const gobot::PhysicsSceneState* single_state = single_world->GetEnvironmentState(environment_index);
        ASSERT_NE(batch_state, nullptr);
        ASSERT_NE(single_state, nullptr);
        ASSERT_EQ(batch_state->robots[0].links.size(), 1);
        const gobot::PhysicsLinkState& batch_link = batch_state->robots[0].links[0];
        const gobot::PhysicsLinkState& single_link = single_state->robots[0].links[0];
        EXPECT_TRUE(batch_link.global_transform.matrix().isApprox(
                single_link.global_transform.matrix(), 1.0e-10));
        EXPECT_TRUE(batch_link.linear_velocity.isApprox(single_link.linear_velocity, 1.0e-10));
        EXPECT_TRUE(batch_link.angular_velocity.isApprox(single_link.angular_velocity, 1.0e-10));
        EXPECT_GT(batch_link.global_transform.translation().x(), 1.0e-3);
    }

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_sets_floating_link_velocity_per_environment) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("velocity_robot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    robot->AddChild(floating_joint);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(1.0);
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});
    floating_joint->AddChild(base);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, robot)) << world->GetLastError();
    ASSERT_TRUE(world->ConfigureEnvironmentBatch(2)) << world->GetLastError();

    ASSERT_TRUE(world->WriteEnvironmentLinkVelocity(
            1,
            "velocity_robot",
            "base",
            {1.0, -2.0, 3.0},
            {0.1, -0.2, 0.3})) << world->GetLastError();

    gobot::PhysicsRobotBatchStepRequest request;
    request.robot_name = "velocity_robot";
    request.base_link = "base";
    request.link_names = {"base"};
    request.ticks = 0;
    gobot::PhysicsRobotBatchStepResult arrays;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    ASSERT_EQ(arrays.base_linear_velocity.size(), 6);
    ASSERT_EQ(arrays.base_angular_velocity.size(), 6);
    EXPECT_NEAR(arrays.base_linear_velocity[0], 0.0, 1.0e-6);
    EXPECT_NEAR(arrays.base_linear_velocity[1], 0.0, 1.0e-6);
    EXPECT_NEAR(arrays.base_linear_velocity[2], 0.0, 1.0e-6);
    EXPECT_NEAR(arrays.base_linear_velocity[3], 1.0, 1.0e-6);
    EXPECT_NEAR(arrays.base_linear_velocity[4], -2.0, 1.0e-6);
    EXPECT_NEAR(arrays.base_linear_velocity[5], 3.0, 1.0e-6);
    EXPECT_NEAR(arrays.base_angular_velocity[3], 0.1, 1.0e-6);
    EXPECT_NEAR(arrays.base_angular_velocity[4], -0.2, 1.0e-6);
    EXPECT_NEAR(arrays.base_angular_velocity[5], 0.3, 1.0e-6);

    const gobot::Quaternion rotated_base(
            gobot::AngleAxis(0.5 * std::acos(-1.0), gobot::Vector3::UnitZ()));
    gobot::PhysicsEnvironmentRobotResetState reset_state;
    reset_state.environment_index = 1;
    reset_state.robot_name = "velocity_robot";
    reset_state.base_link_name = "base";
    reset_state.base_orientation = rotated_base;
    reset_state.base_angular_velocity = {0.4, 0.5, -0.6};
    ASSERT_TRUE(world->ResetEnvironmentRobotStates({reset_state})) << world->GetLastError();
    request.ticks = 0;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    EXPECT_NEAR(arrays.base_angular_velocity[3], 0.4, 1.0e-6);
    EXPECT_NEAR(arrays.base_angular_velocity[4], 0.5, 1.0e-6);
    EXPECT_NEAR(arrays.base_angular_velocity[5], -0.6, 1.0e-6);

    ASSERT_TRUE(world->WriteEnvironmentLinkVelocity(
            1,
            "velocity_robot",
            "base",
            {1.0, -2.0, 3.0},
            {-0.7, 0.8, 0.9})) << world->GetLastError();
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    EXPECT_NEAR(arrays.base_angular_velocity[3], -0.7, 1.0e-6);
    EXPECT_NEAR(arrays.base_angular_velocity[4], 0.8, 1.0e-6);
    EXPECT_NEAR(arrays.base_angular_velocity[5], 0.9, 1.0e-6);

    request.ticks = 1;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    const gobot::RealType pre_forward_x = arrays.base_position[3];
    request.ticks = 0;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    EXPECT_GT(arrays.base_position[3], pre_forward_x);

    EXPECT_FALSE(world->WriteEnvironmentLinkVelocity(
            1,
            "velocity_robot",
            "missing",
            gobot::Vector3::Zero(),
            gobot::Vector3::Zero()));
    EXPECT_FALSE(world->GetLastError().empty());

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_batch_step_applies_named_link_overrides_and_wrenches) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("override_robot");

    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    robot->AddChild(floating_joint);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(1.0);
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});
    floating_joint->AddChild(base);

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.01;
    settings.gravity = gobot::Vector3::Zero();
    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu, settings);
    ASSERT_TRUE(BuildWorldFromScene(world, robot)) << world->GetLastError();
    ASSERT_TRUE(world->ConfigureEnvironmentBatch(2)) << world->GetLastError();

    gobot::PhysicsRobotBatchStepRequest request;
    request.robot_name = "override_robot";
    request.base_link = "base";
    request.link_names = {"base"};
    request.override_link_names = {"base"};
    request.link_mass_delta = {0.0, 1.0};
    request.link_center_of_mass_offset = {0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0};
    request.external_wrench_link = "base";
    request.external_force = {1.0, 0.0, 0.0,
                              1.0, 0.0, 0.0};
    request.external_torque.assign(6, 0.0);
    request.ticks = 10;
    request.worker_count = 2;

    gobot::PhysicsRobotBatchStepResult arrays;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    ASSERT_EQ(arrays.base_linear_velocity.size(), 6);
    const gobot::RealType unit_mass_velocity = arrays.base_linear_velocity[0];
    const gobot::RealType double_mass_velocity = arrays.base_linear_velocity[3];
    EXPECT_GT(unit_mass_velocity, 0.0);
    EXPECT_GT(double_mass_velocity, 0.0);
    EXPECT_NEAR(unit_mass_velocity / double_mass_velocity, 2.0, 0.05);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_batch_step_accepts_zero_pd_gains_and_rejects_invalid_gains) {
#ifdef GOBOT_HAS_MUJOCO
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("gain_robot");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetMass(10.0);
    base->SetInertiaDiagonal({1.0, 1.0, 1.0});
    robot->AddChild(base);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("hinge");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("tip");
    joint->SetAxis({0.0, 0.0, 1.0});
    joint->SetLowerLimit(-2.0);
    joint->SetUpperLimit(2.0);
    joint->SetDriveMode(gobot::JointDriveMode::Position);
    joint->SetDriveStiffness(30.0);
    joint->SetDriveDamping(1.0);
    base->AddChild(joint);

    auto* tip = gobot::Object::New<gobot::Link3D>();
    tip->SetName("tip");
    tip->SetMass(1.0);
    tip->SetCenterOfMass({0.2, 0.0, 0.0});
    tip->SetInertiaDiagonal({0.01, 0.05, 0.05});
    joint->AddChild(tip);

    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.005;
    settings.gravity = gobot::Vector3::Zero();
    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu, settings);
    ASSERT_TRUE(BuildWorldFromScene(world, robot)) << world->GetLastError();
    ASSERT_TRUE(world->ConfigureEnvironmentBatch(2)) << world->GetLastError();

    gobot::PhysicsRobotBatchStepRequest request;
    request.robot_name = "gain_robot";
    request.base_link = "base";
    request.joint_names = {"hinge"};
    request.link_names = {"base", "tip"};
    request.target_positions = {1.0, 1.0};
    request.joint_position_stiffness = {0.0, 30.0};
    request.joint_velocity_damping = {0.0, 1.0};
    request.ticks = 100;

    gobot::PhysicsRobotBatchStepResult arrays;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    ASSERT_EQ(arrays.joint_position.size(), 2);
    EXPECT_NEAR(arrays.joint_position[0], 0.0, 1.0e-8);
    EXPECT_GT(arrays.joint_position[1], 0.05);

    request.joint_position_stiffness[0] = -1.0;
    EXPECT_FALSE(world->StepRobotBatch(request, arrays));
    EXPECT_NE(world->GetLastError().find("finite and non-negative"), std::string::npos);

    gobot::Object::Delete(robot);
#endif
}

TEST(TestPhysicsServer, mujoco_batch_step_reports_named_contact_history) {
#ifdef GOBOT_HAS_MUJOCO
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");
    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->AddBox({0.0, 0.0, -0.05}, {4.0, 4.0, 0.1});
    root->AddChild(terrain);

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("contact_robot");
    auto* floating_joint = gobot::Object::New<gobot::Joint3D>();
    floating_joint->SetName("floating_base_joint");
    floating_joint->SetJointType(gobot::JointType::Floating);
    floating_joint->SetChildLink("base");
    robot->AddChild(floating_joint);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetPosition({0.0, 0.0, 0.08});
    base->SetMass(1.0);
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});
    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName("base_collision");
    auto shape = gobot::MakeRef<gobot::BoxShape3D>();
    shape->SetSize({0.2, 0.2, 0.2});
    collision->SetShape(shape);
    base->AddChild(collision);
    floating_joint->AddChild(base);
    root->AddChild(robot);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, root)) << world->GetLastError();
    ASSERT_TRUE(world->ConfigureEnvironmentBatch(1)) << world->GetLastError();

    gobot::PhysicsRobotBatchStepRequest request;
    request.robot_name = "contact_robot";
    request.base_link = "base";
    request.link_names = {"base"};
    request.override_shape_names = {"base_collision"};
    request.shape_friction = {0.8, 0.02, 0.001};
    request.shape_friction_enabled = {1};
    request.contact_shape_groups = {{"base_group", {"base_collision"}, 0.0}};
    request.collect_contact_history = true;
    request.ticks = 8;

    gobot::PhysicsRobotBatchStepResult arrays;
    ASSERT_TRUE(world->StepRobotBatch(request, arrays)) << world->GetLastError();
    ASSERT_EQ(arrays.shape_names.size(), 1);
    EXPECT_EQ(arrays.shape_names[0], "base_collision");
    ASSERT_EQ(arrays.shape_paths.size(), 1);
    EXPECT_NE(arrays.shape_paths[0].find("base_collision"), std::string::npos);
    ASSERT_EQ(arrays.shape_stable_ids.size(), 1);
    EXPECT_NE(arrays.shape_stable_ids[0], 0);
    ASSERT_EQ(arrays.link_contact_tick_count.size(), 1);
    EXPECT_GT(arrays.link_contact_tick_count[0], 0);
    ASSERT_EQ(arrays.contact_shape_group_names, std::vector<std::string>{"base_group"});
    ASSERT_EQ(arrays.contact_shape_group_tick_count.size(), 1);
    EXPECT_GT(arrays.contact_shape_group_tick_count[0], 0);
    EXPECT_LE(arrays.contact_shape_group_tick_count[0], request.ticks);
    ASSERT_EQ(arrays.contact_count.size(), 1);
    EXPECT_GT(arrays.contact_count[0], 0);
    ASSERT_GE(arrays.contact_shape_index.size(), 2);
    EXPECT_TRUE(arrays.contact_shape_index[0] == 0 || arrays.contact_shape_index[1] == 0);
    ASSERT_GE(arrays.contact_force.size(), 3);
    ASSERT_GE(arrays.contact_tangent_force.size(), 3);
    ASSERT_GE(arrays.contact_torque.size(), 3);
    ASSERT_GE(arrays.contact_normal.size(), 3);
    ASSERT_GE(arrays.contact_normal_force.size(), 1);
    ASSERT_GE(arrays.contact_normal_impulse.size(), 1);
    const gobot::Vector3 contact_force(
            arrays.contact_force[0], arrays.contact_force[1], arrays.contact_force[2]);
    const gobot::Vector3 contact_normal(
            arrays.contact_normal[0], arrays.contact_normal[1], arrays.contact_normal[2]);
    const gobot::Vector3 tangent_force(
            arrays.contact_tangent_force[0],
            arrays.contact_tangent_force[1],
            arrays.contact_tangent_force[2]);
    const gobot::Vector3 contact_torque(
            arrays.contact_torque[0], arrays.contact_torque[1], arrays.contact_torque[2]);
    EXPECT_TRUE(contact_force.allFinite());
    EXPECT_TRUE(contact_torque.allFinite());
    EXPECT_GT(arrays.contact_normal_force[0], 0.0);
    EXPECT_NEAR(tangent_force.dot(contact_normal), 0.0, 1.0e-8);
    EXPECT_TRUE(tangent_force.isApprox(
            contact_force - contact_normal * arrays.contact_normal_force[0],
            1.0e-8));
    EXPECT_NEAR(
            arrays.contact_normal_impulse[0],
            arrays.contact_normal_force[0] * world->GetSettings().fixed_time_step,
            1.0e-10);

    const gobot::PhysicsSceneState* runtime_state = world->GetEnvironmentState(0);
    ASSERT_NE(runtime_state, nullptr);
    const auto runtime_contact = std::ranges::find_if(
            runtime_state->contacts,
            [](const gobot::PhysicsContactState& contact) {
                return contact.shape_name == "base_collision";
            });
    ASSERT_NE(runtime_contact, runtime_state->contacts.end());
    EXPECT_NE(runtime_contact->shape_stable_id, 0);
    EXPECT_NE(runtime_contact->other_shape_stable_id, 0);
    EXPECT_NE(runtime_contact->shape_path.find("base_collision"), std::string::npos);
    EXPECT_NE(runtime_contact->other_shape_path.find("terrain"), std::string::npos);
    EXPECT_NE(runtime_contact->other_shape_name.find("gobot_terrain_box_"),
              std::string::npos);

    gobot::Object::Delete(root);
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

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, robot)) << world->GetLastError();
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

TEST(TestPhysicsServer, mujoco_compiles_robot_and_loose_convex_mesh_shapes) {
#ifdef GOBOT_HAS_MUJOCO
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto mesh = CreateTetrahedronMesh();
    auto convex_shape = gobot::MakeRef<gobot::ConvexMeshShape3D>();
    convex_shape->SetMesh(mesh);

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("mesh_robot");
    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("base");
    link->SetPosition({1.0, 2.0, 3.0});
    link->SetMass(1.0);
    link->SetCenterOfMass({0.1, 0.1, 0.1});
    link->SetInertiaDiagonal({0.1, 0.1, 0.1});
    auto* robot_collision = gobot::Object::New<gobot::CollisionShape3D>();
    robot_collision->SetName("robot_convex");
    robot_collision->SetPosition({0.1, 0.2, 0.3});
    auto robot_material = gobot::MakeRef<gobot::PhysicsMaterial3D>();
    robot_material->SetSlidingFriction(0.7);
    robot_material->SetTorsionalFriction(0.02);
    robot_material->SetRollingFriction(0.003);
    robot_collision->SetPhysicsMaterial(robot_material);
    robot_collision->SetContactOffset(0.01);
    robot_collision->SetRestOffset(0.008);
    robot_collision->SetShape(convex_shape);
    link->AddChild(robot_collision);
    robot->AddChild(link);
    root->AddChild(robot);

    auto* loose_collision = gobot::Object::New<gobot::CollisionShape3D>();
    loose_collision->SetName("loose_convex");
    loose_collision->SetPosition({-2.0, 0.0, 0.0});
    loose_collision->SetShape(convex_shape);
    root->AddChild(loose_collision);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, root)) << world->GetLastError();

    const gobot::PhysicsSceneArtifact* artifact = world->GetSceneArtifact();
    ASSERT_NE(artifact, nullptr);
    EXPECT_EQ(artifact->ngeom, 2);
    EXPECT_EQ(CountOccurrences(artifact->content, "<mesh name="), 2);
    EXPECT_EQ(CountOccurrences(artifact->content, "type=\"mesh\""), 2);
    EXPECT_NE(artifact->content.find("vertex=\""), std::string::npos);
    EXPECT_NE(artifact->content.find("face=\""), std::string::npos);
    EXPECT_NE(artifact->content.find("name=\"mesh_robot_robot_convex_mesh\""),
              std::string::npos);
    EXPECT_NE(artifact->content.find("mesh=\"mesh_robot_robot_convex_mesh\""),
              std::string::npos);
    EXPECT_NE(artifact->content.find("name=\"gobot_loose_box_0_mesh\""),
              std::string::npos);
    EXPECT_NE(artifact->content.find("mesh=\"gobot_loose_box_0_mesh\""),
              std::string::npos);

    gobot::Object::Delete(root);
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

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, root)) << world->GetLastError();
    world->Step(0.002);
    EXPECT_EQ(world->GetSceneSnapshot().total_terrain_count, 1);
    EXPECT_EQ(world->GetSceneSnapshot().terrains[0].heightfields.size(), 1);

    const gobot::PhysicsSceneArtifact* artifact = world->GetSceneArtifact();
    ASSERT_NE(artifact, nullptr);
    EXPECT_EQ(artifact->schema_version, gobot::MuJoCoSceneCompiler::kArtifactSchemaVersion);
    EXPECT_EQ(artifact->backend, gobot::PhysicsBackendType::MuJoCoCpu);
    EXPECT_EQ(artifact->format, "mjcf");
    EXPECT_FALSE(artifact->content_digest.empty());
    EXPECT_FALSE(artifact->backend_version.empty());
    EXPECT_NE(artifact->content.find("<hfield"), std::string::npos);
    EXPECT_NE(artifact->content.find("elevation="), std::string::npos);
    EXPECT_EQ(artifact->nhfield, 1);
    ASSERT_EQ(artifact->robot_names, std::vector<std::string>{"terrain_bot"});
    ASSERT_EQ(artifact->robot_prefixes, std::vector<std::string>{"terrain_bot_"});
    ASSERT_EQ(artifact->terrain_geom_groups, std::vector<std::int32_t>{5});

    gobot::PhysicsSceneArtifact recompiled_artifact;
    std::string compile_error;
    ASSERT_TRUE(gobot::PhysicsServer::CompileSceneArtifact(
            gobot::PhysicsBackendType::MuJoCoCpu,
            world->GetSceneSnapshot(),
            world->GetSettings(),
            &recompiled_artifact,
            &compile_error)) << compile_error;
    EXPECT_EQ(recompiled_artifact.content_digest, artifact->content_digest);
    EXPECT_EQ(recompiled_artifact.content, artifact->content);
    EXPECT_EQ(recompiled_artifact.nq, artifact->nq);
    EXPECT_EQ(recompiled_artifact.nv, artifact->nv);
    EXPECT_EQ(recompiled_artifact.nu, artifact->nu);
    EXPECT_EQ(recompiled_artifact.terrain_geom_groups, artifact->terrain_geom_groups);

    gobot::Object::Delete(root);
#endif
}

TEST(TestPhysicsServer, mujoco_terrain_raycast_ignores_robot_collision_geoms) {
#ifdef GOBOT_HAS_MUJOCO
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");
    terrain->AddBox({0.0, 0.0, -0.05}, {4.0, 4.0, 0.1});
    root->AddChild(terrain);

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("raycast_bot");
    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetPosition({0.0, 0.0, 0.5});
    base->SetMass(1.0);
    base->SetCenterOfMass({0.0, 0.0, 0.0});
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});

    auto* collision = gobot::Object::New<gobot::CollisionShape3D>();
    collision->SetName("base_collision");
    auto shape = gobot::MakeRef<gobot::BoxShape3D>();
    shape->SetSize({0.5, 0.5, 0.5});
    collision->SetShape(shape);
    base->AddChild(collision);
    robot->AddChild(base);
    root->AddChild(robot);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, root)) << world->GetLastError();

    const gobot::PhysicsRaycastHit hit = world->RaycastTerrain({
            {0.0, 0.0, 2.0},
            {0.0, 0.0, -1.0},
            4.0
    });
    ASSERT_TRUE(hit.hit);
    EXPECT_NEAR(hit.point.z(), 0.0, 1.0e-6);
    EXPECT_GT(hit.distance, 1.9);
    EXPECT_NE(hit.terrain_name.find("gobot_terrain_box_0"), std::string::npos);

    gobot::Object::Delete(root);
#endif
}

TEST(TestPhysicsServer, mujoco_heightfield_raycast_uses_visible_heights_not_palette_values) {
#ifdef GOBOT_HAS_MUJOCO
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetName("terrain");

    gobot::TerrainHeightField heightfield;
    heightfield.size = {2.0, 2.0};
    heightfield.rows = 2;
    heightfield.cols = 2;
    heightfield.heights = {0.2, 0.2, 0.2, 0.2};
    heightfield.normalized_elevation = {0.0, 1.0, 0.0, 1.0};
    heightfield.z_offset = -0.1;
    terrain->AddHeightField(heightfield);
    root->AddChild(terrain);

    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("hfield_probe_bot");
    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetPosition({4.0, 4.0, 1.0});
    base->SetMass(1.0);
    base->SetCenterOfMass({0.0, 0.0, 0.0});
    base->SetInertiaDiagonal({0.01, 0.01, 0.01});
    robot->AddChild(base);
    root->AddChild(robot);

    gobot::PhysicsServer physics_server;
    gobot::Ref<gobot::PhysicsWorld> world = physics_server.CreateWorld(gobot::PhysicsBackendType::MuJoCoCpu);
    ASSERT_TRUE(BuildWorldFromScene(world, root)) << world->GetLastError();

    const gobot::PhysicsRaycastHit hit = world->RaycastTerrain({
            {0.0, 0.0, 1.0},
            {0.0, 0.0, -1.0},
            2.0
    });
    ASSERT_TRUE(hit.hit);
    EXPECT_NEAR(hit.point.z(), 0.1, 1.0e-5);
    EXPECT_NE(hit.terrain_name.find("hfield"), std::string::npos);

    gobot::Object::Delete(root);
#endif
}
