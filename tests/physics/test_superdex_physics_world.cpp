#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gobot/physics/backends/superdex_conversions.hpp>
#include <gobot/physics/physics_server.hpp>

#ifdef GOBOT_HAS_SUPERDEX
#include <gobot/physics/backends/superdex_physics_world.hpp>
#endif

namespace {

gobot::PhysicsSceneSnapshot MakeRigidDropSnapshot() {
    gobot::PhysicsSceneSnapshot snapshot;
    gobot::PhysicsRobotSnapshot robot;
    robot.name = "drop_box";
    robot.scene_path = "/world/drop_box";
    robot.stable_id = 1;
    robot.standalone_rigid_body = true;

    gobot::PhysicsLinkSnapshot link;
    link.name = "drop_box";
    link.scene_path = robot.scene_path;
    link.stable_id = robot.stable_id;
    link.mass = 1.0;
    link.inertia_diagonal = {0.01, 0.01, 0.01};
    link.global_transform = gobot::Affine3::Identity();
    link.global_transform.translation() = gobot::Vector3{0.0, 0.0, 1.0};

    gobot::PhysicsShapeSnapshot shape;
    shape.name = "drop_box_collision";
    shape.scene_path = "/world/drop_box/drop_box_collision";
    shape.stable_id = 2;
    shape.type = gobot::PhysicsShapeType::Box;
    shape.global_transform = link.global_transform;
    shape.box_size = {0.1, 0.1, 0.1};
    shape.collision_layer = 1;
    shape.collision_mask = 1;
    link.collision_shapes.push_back(std::move(shape));
    robot.links.push_back(std::move(link));
    snapshot.robots.push_back(std::move(robot));
    snapshot.total_link_count = 1;
    return snapshot;
}

gobot::PhysicsSceneSnapshot MakeRestingBoxSnapshot() {
    gobot::PhysicsSceneSnapshot snapshot = MakeRigidDropSnapshot();
    gobot::PhysicsLinkSnapshot& link = snapshot.robots[0].links[0];
    link.global_transform.translation() = gobot::Vector3{0.0, 0.0, 0.051};
    link.collision_shapes[0].global_transform = link.global_transform;

    gobot::PhysicsTerrainSnapshot terrain;
    terrain.name = "ground";
    terrain.scene_path = "/world/ground";
    terrain.stable_id = 3;
    terrain.collision_layer = 1;
    terrain.collision_mask = 1;
    gobot::PhysicsTerrainBoxSnapshot ground;
    ground.size = {2.0, 2.0, 0.1};
    ground.global_transform.translation() = gobot::Vector3{0.0, 0.0, -0.05};
    terrain.boxes.push_back(std::move(ground));
    snapshot.terrains.push_back(std::move(terrain));
    return snapshot;
}

gobot::PhysicsSceneSnapshot MakeTetrahedronSnapshot() {
    gobot::PhysicsSceneSnapshot snapshot;
    gobot::PhysicsDeformableSnapshot deformable;
    deformable.name = "tetrahedron";
    deformable.scene_path = "/world/tetrahedron";
    deformable.stable_id = 41;
    deformable.global_transform = gobot::Affine3::Identity();
    deformable.global_transform.translation() = gobot::Vector3{0.0, 0.0, 0.5};
    deformable.vertices = {
            {0.0, 0.0, 0.0},
            {0.05, 0.0, 0.0},
            {0.0, 0.05, 0.0},
            {0.0, 0.0, 0.05},
    };
    deformable.tetrahedra = {0, 1, 2, 3};
    deformable.surface_triangles = {
            0, 2, 1,
            0, 1, 3,
            1, 2, 3,
            2, 0, 3,
    };
    deformable.model = 0;
    deformable.density = 1000.0;
    deformable.young_modulus = 10000.0;
    deformable.poisson_ratio = 0.3;
    deformable.collision_layer = 1;
    deformable.collision_mask = 1;
    snapshot.deformables.push_back(std::move(deformable));
    snapshot.total_deformable_count = 1;
    return snapshot;
}

gobot::PhysicsShapeSnapshot MakeBoxShape(const std::string& name,
                                         std::uint64_t stable_id,
                                         const gobot::Affine3& transform,
                                         const gobot::Vector3& size) {
    gobot::PhysicsShapeSnapshot shape;
    shape.name = name;
    shape.scene_path = "/world/hinge/" + name;
    shape.stable_id = stable_id;
    shape.type = gobot::PhysicsShapeType::Box;
    shape.global_transform = transform;
    shape.box_size = size;
    shape.collision_layer = 1;
    shape.collision_mask = 1;
    return shape;
}

gobot::PhysicsSceneSnapshot MakeDrivenHingeSnapshot() {
    gobot::PhysicsSceneSnapshot snapshot;
    gobot::PhysicsRobotSnapshot robot;
    robot.name = "hinge";
    robot.scene_path = "/world/hinge";
    robot.stable_id = 50;

    gobot::PhysicsLinkSnapshot base;
    base.name = "base";
    base.scene_path = "/world/hinge/base";
    base.stable_id = 51;
    base.mass = 1.0;
    base.inertia_diagonal = {0.01, 0.01, 0.01};
    base.collision_shapes.push_back(
            MakeBoxShape("base_collision", 52, base.global_transform, {0.1, 0.1, 0.1}));

    gobot::PhysicsLinkSnapshot tip;
    tip.name = "tip";
    tip.scene_path = "/world/hinge/tip";
    tip.stable_id = 53;
    tip.mass = 0.2;
    tip.inertia_diagonal = {0.001, 0.001, 0.001};
    tip.global_transform.translation() = gobot::Vector3{0.15, 0.0, 0.0};
    tip.collision_shapes.push_back(
            MakeBoxShape("tip_collision", 54, tip.global_transform, {0.2, 0.04, 0.04}));

    gobot::PhysicsJointSnapshot joint;
    joint.name = "hinge_joint";
    joint.scene_path = "/world/hinge/hinge_joint";
    joint.stable_id = 55;
    joint.parent_link = "base";
    joint.child_link = "tip";
    joint.axis = gobot::Vector3::UnitZ();
    joint.lower_limit = -1.0;
    joint.upper_limit = 1.0;
    joint.effort_limit = 20.0;
    joint.velocity_limit = 10.0;
    joint.drive_stiffness = 30.0;
    joint.drive_damping = 2.0;
    joint.joint_type = static_cast<int>(gobot::JointType::Revolute);

    robot.links.push_back(std::move(base));
    robot.links.push_back(std::move(tip));
    robot.joints.push_back(std::move(joint));
    snapshot.robots.push_back(std::move(robot));
    snapshot.total_link_count = 2;
    snapshot.total_joint_count = 1;
    return snapshot;
}

gobot::PhysicsSceneSnapshot MakeShellSnapshot() {
    gobot::PhysicsSceneSnapshot snapshot;
    gobot::PhysicsDeformableSnapshot deformable;
    deformable.name = "shell";
    deformable.scene_path = "/world/shell";
    deformable.stable_id = 61;
    deformable.global_transform.translation() = gobot::Vector3{0.0, 0.0, 0.5};
    deformable.vertices = {
            {-0.05, -0.05, 0.0},
            {0.05, -0.05, 0.0},
            {0.05, 0.05, 0.0},
            {-0.05, 0.05, 0.0},
    };
    deformable.surface_triangles = {0, 1, 2, 0, 2, 3};
    deformable.model = 1;
    deformable.density = 1000.0;
    deformable.young_modulus = 10000.0;
    deformable.poisson_ratio = 0.3;
    deformable.thickness = 0.001;
    deformable.bending_stiffness = 1.0e-4;
    deformable.self_collision_enabled = true;
    deformable.collision_layer = 1;
    deformable.collision_mask = 1;
    snapshot.deformables.push_back(std::move(deformable));
    snapshot.total_deformable_count = 1;
    return snapshot;
}

} // namespace

TEST(TestSuperDexConversions, maps_vectors_and_frames_with_a_stable_right_handed_basis) {
    const gobot::Vector3 value{1.0, 2.0, 3.0};
    EXPECT_TRUE(gobot::superdex::ToMochiVector(value).isApprox(
            gobot::Vector3{1.0, 3.0, -2.0}, CMP_EPSILON));
    EXPECT_TRUE(gobot::superdex::FromMochiVector(
                        gobot::superdex::ToMochiVector(value))
                        .isApprox(value, CMP_EPSILON));
    EXPECT_NEAR(gobot::superdex::GobotToMochiBasis().determinant(), 1.0, CMP_EPSILON);

    gobot::Affine3 frame = gobot::Affine3::Identity();
    frame.translation() = value;
    frame.linear() = gobot::AngleAxis(0.37, gobot::Vector3{1.0, 2.0, 3.0}.normalized())
                             .toRotationMatrix();
    const gobot::Affine3 round_trip = gobot::superdex::FromMochiFrame(
            gobot::superdex::ToMochiFrame(frame));
    EXPECT_TRUE(round_trip.matrix().isApprox(frame.matrix(), 1.0e-5));
}

TEST(TestSuperDexConversions, maps_supported_joints_and_rejects_planar) {
    EXPECT_EQ(gobot::superdex::ToJointKind(gobot::JointType::Fixed),
              gobot::superdex::JointKind::Hard);
    EXPECT_EQ(gobot::superdex::ToJointKind(gobot::JointType::Revolute),
              gobot::superdex::JointKind::Revolute);
    EXPECT_EQ(gobot::superdex::ToJointKind(gobot::JointType::Continuous),
              gobot::superdex::JointKind::Revolute);
    EXPECT_EQ(gobot::superdex::ToJointKind(gobot::JointType::Prismatic),
              gobot::superdex::JointKind::Prismatic);
    EXPECT_EQ(gobot::superdex::ToJointKind(gobot::JointType::Floating),
              gobot::superdex::JointKind::Free);
    EXPECT_EQ(gobot::superdex::ToJointKind(gobot::JointType::Planar),
              gobot::superdex::JointKind::Unsupported);
}

TEST(TestSuperDexConversions, rotates_and_changes_basis_for_link_inertia) {
    gobot::PhysicsLinkSnapshot link;
    link.mass = 2.0;
    link.inertia_diagonal = {1.0, 2.0, 3.0};
    link.inertia_orientation = gobot::Quaternion(
            gobot::AngleAxis(0.5 * Math_PI, gobot::Vector3::UnitZ()));
    const gobot::Matrix3 converted = gobot::superdex::ToMochiInertiaTensor(link);
    const gobot::Matrix3 expected = (gobot::Vector3{2.0, 3.0, 1.0}).asDiagonal();
    EXPECT_TRUE(converted.isApprox(expected, 1.0e-5));
}

TEST(TestSuperDexConversions, requires_bilateral_collision_masks) {
    EXPECT_TRUE(gobot::superdex::ShouldEnableContact(0b0010, 0b0100, 0b0100, 0b0010));
    EXPECT_FALSE(gobot::superdex::ShouldEnableContact(0b0010, 0b0100, 0b0100, 0b1000));
    EXPECT_EQ(gobot::superdex::MakeContactLayerName(0x12U, 0x34U),
              "gobot_00000012_00000034");
}

TEST(TestSuperDexConversions, clamps_shell_contact_radius_to_authored_limits) {
    const std::vector<gobot::Vector3> vertices = {
            {0.0, 0.0, 0.0},
            {0.02, 0.0, 0.0},
            {0.0, 0.02, 0.0},
    };
    const std::vector<std::uint32_t> triangles = {0, 1, 2};
    EXPECT_NEAR(
            gobot::superdex::ComputeShellContactRadius(vertices, triangles, 0.001),
            0.01,
            CMP_EPSILON);
    EXPECT_NEAR(
            gobot::superdex::ComputeShellContactRadius(vertices, triangles, 0.02),
            0.01,
            CMP_EPSILON);
}

TEST(TestSuperDexConversions, requests_quadrature_for_coarse_shell_contact_meshes) {
    const std::vector<gobot::Vector3> coarse_vertices = {
            {0.0, 0.0, 0.0},
            {0.04, 0.0, 0.0},
            {0.0, 0.04, 0.0},
    };
    const std::vector<std::uint32_t> triangle = {0, 1, 2};
    EXPECT_TRUE(gobot::superdex::RequiresShellContactQuadrature(
            coarse_vertices, triangle, 0.01));

    const std::vector<gobot::Vector3> resolved_vertices = {
            {0.0, 0.0, 0.0},
            {0.01, 0.0, 0.0},
            {0.0, 0.01, 0.0},
    };
    EXPECT_FALSE(gobot::superdex::RequiresShellContactQuadrature(
            resolved_vertices, triangle, 0.01));
    EXPECT_FALSE(gobot::superdex::RequiresShellContactQuadrature(
            coarse_vertices, triangle, 0.0));
}

#ifdef GOBOT_HAS_SUPERDEX

TEST(TestSuperDexPhysicsWorld, registers_as_an_experimental_cpu_backend) {
    gobot::PhysicsServer server;
    const gobot::PhysicsBackendInfo info =
            server.GetBackendInfo(gobot::PhysicsBackendType::SuperDex);
    EXPECT_TRUE(info.available);
    EXPECT_TRUE(info.cpu);
    EXPECT_TRUE(info.robotics_focused);
    EXPECT_NE(info.status.find("Experimental"), std::string::npos);
}

TEST(TestSuperDexPhysicsWorld, rejects_cuda_execution_in_a_cpu_only_build) {
    gobot::PhysicsServer server;
    const gobot::PhysicsBackendInfo info =
            server.GetBackendInfo(gobot::PhysicsBackendType::SuperDex);
    if (info.gpu) {
        GTEST_SKIP() << "This build contains the SuperDex CUDA linear solver.";
    }

    gobot::PhysicsWorldSettings settings;
    settings.superdex_solver.execution_mode = gobot::SuperDexExecutionMode::Cuda;
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    EXPECT_FALSE(world->Build(MakeRigidDropSnapshot()));
    EXPECT_NE(world->GetLastError().find("only the CPU solver"), std::string::npos);
}

TEST(TestSuperDexPhysicsWorld, rigid_body_falls_and_checkpoint_replays_exactly) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.002;
    settings.gravity = {0.0, 0.0, -9.81};
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    gobot::PhysicsSceneSnapshot snapshot = MakeRigidDropSnapshot();
    ASSERT_EQ(snapshot.robots.size(), 1u);
    ASSERT_EQ(snapshot.robots[0].links.size(), 1u);
    ASSERT_TRUE(world->Build(std::move(snapshot))) << world->GetLastError();

    const gobot::PhysicsSceneState& initial = world->GetSceneState();
    ASSERT_EQ(initial.robots.size(), 1u);
    ASSERT_EQ(initial.robots[0].links.size(), 1u);
    const gobot::RealType initial_height =
            initial.robots[0].links[0].global_transform.translation().z();
    for (int step = 0; step < 5; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    }
    const gobot::RealType checkpoint_height =
            world->GetSceneState().robots[0].links[0].global_transform.translation().z();
    EXPECT_LT(checkpoint_height, initial_height);
    gobot::Ref<gobot::PhysicsRuntimeCheckpoint> checkpoint = world->CaptureCheckpoint();
    ASSERT_TRUE(checkpoint.IsValid());
    EXPECT_EQ(checkpoint->GetSchemaVersion(), 2u);

    for (int step = 0; step < 5; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    }
    EXPECT_LT(world->GetSceneState().robots[0].links[0].global_transform.translation().z(),
              checkpoint_height);
    ASSERT_TRUE(world->RestoreCheckpoint(checkpoint)) << world->GetLastError();
    EXPECT_NEAR(world->GetSceneState().robots[0].links[0].global_transform.translation().z(),
                checkpoint_height,
                1.0e-6);

    const gobot::PhysicsSolverDiagnostics diagnostics = world->GetSolverDiagnostics();
    EXPECT_EQ(diagnostics.execution_device, "cpu");
    EXPECT_FALSE(diagnostics.device_native);
    EXPECT_FALSE(diagnostics.graph_capture);
    EXPECT_TRUE(std::isfinite(diagnostics.residual_norm));
}

TEST(TestSuperDexPhysicsWorld, builds_supported_rigid_shapes_and_applies_link_force) {
    const std::array<gobot::PhysicsShapeType, 5> shape_types = {
            gobot::PhysicsShapeType::Box,
            gobot::PhysicsShapeType::Sphere,
            gobot::PhysicsShapeType::Cylinder,
            gobot::PhysicsShapeType::Capsule,
            gobot::PhysicsShapeType::Mesh,
    };

    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.001;
    settings.gravity = gobot::Vector3::Zero();
    for (const gobot::PhysicsShapeType shape_type : shape_types) {
        SCOPED_TRACE(static_cast<int>(shape_type));
        gobot::PhysicsSceneSnapshot snapshot = MakeRigidDropSnapshot();
        gobot::PhysicsShapeSnapshot& shape =
                snapshot.robots[0].links[0].collision_shapes[0];
        shape.type = shape_type;
        shape.radius = 0.03;
        shape.height = 0.08;
        if (shape_type == gobot::PhysicsShapeType::Mesh) {
            shape.vertices = {
                    {-0.05, -0.05, -0.05},
                    {0.05, -0.05, -0.05},
                    {0.0, 0.05, -0.05},
                    {0.0, 0.0, 0.05},
            };
            shape.indices = {
                    0, 2, 1,
                    0, 1, 3,
                    1, 2, 3,
                    2, 0, 3,
            };
        }

        gobot::Ref<gobot::PhysicsWorld> world =
                server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
        ASSERT_TRUE(world.IsValid());
        ASSERT_TRUE(world->Build(std::move(snapshot))) << world->GetLastError();
        ASSERT_TRUE(world->SetLinkExternalForce(
                "drop_box", "drop_box", {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}));
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
        const gobot::PhysicsLinkState& state =
                world->GetSceneState().robots[0].links[0];
        EXPECT_TRUE(state.global_transform.matrix().allFinite());
        EXPECT_TRUE(state.linear_velocity.allFinite());
        EXPECT_GT(state.linear_velocity.x(), 0.0);
        world.Reset();
    }
}

TEST(TestSuperDexPhysicsWorld, resting_mesh_box_has_no_horizontal_sdf_drift) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.002;
    settings.gravity = {0.0, 0.0, -9.81};
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeRestingBoxSnapshot())) << world->GetLastError();

    const gobot::Affine3 initial =
            world->GetSceneState().robots[0].links[0].global_transform;
    gobot::RealType peak_penetration = 0.0;
    for (int step = 0; step < 100; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
        for (const gobot::PhysicsContactState& contact :
             world->GetSceneState().contacts) {
            peak_penetration = std::max(
                    peak_penetration, std::max<gobot::RealType>(0.0, -contact.distance));
        }
    }

    const gobot::Affine3 settled =
            world->GetSceneState().robots[0].links[0].global_transform;
    EXPECT_LT((settled.translation() - initial.translation()).head<2>().norm(),
              1.0e-5);
    EXPECT_TRUE(settled.linear().isApprox(initial.linear(), 1.0e-5));
    EXPECT_LE(peak_penetration, 1.0e-3);
}

TEST(TestSuperDexPhysicsWorld, rejects_planar_joints_and_legacy_coupling) {
    gobot::PhysicsServer server;
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex);
    ASSERT_TRUE(world.IsValid());

    gobot::PhysicsSceneSnapshot planar;
    gobot::PhysicsRobotSnapshot robot;
    robot.name = "planar_robot";
    robot.scene_path = "/world/planar_robot";
    gobot::PhysicsLinkSnapshot link;
    link.name = "base";
    link.scene_path = "/world/planar_robot/base";
    robot.links.push_back(std::move(link));
    gobot::PhysicsJointSnapshot joint;
    joint.name = "planar";
    joint.scene_path = "/world/planar_robot/planar";
    joint.child_link = "base";
    joint.joint_type = static_cast<int>(gobot::JointType::Planar);
    robot.joints.push_back(std::move(joint));
    planar.robots.push_back(std::move(robot));
    EXPECT_FALSE(world->Build(std::move(planar)));
    EXPECT_NE(world->GetLastError().find("Planar"), std::string::npos);

    gobot::PhysicsSceneSnapshot coupled;
    gobot::PhysicsCouplingSnapshot coupling;
    coupling.name = "legacy";
    coupling.scene_path = "/world/legacy";
    coupling.stable_id = 9;
    coupled.couplings.push_back(std::move(coupling));
    EXPECT_FALSE(world->Build(std::move(coupled)));
    EXPECT_NE(world->GetLastError().find("unified solver"), std::string::npos);
}

TEST(TestSuperDexPhysicsWorld, rejects_invalid_deformable_models_and_connectivity) {
    gobot::PhysicsServer server;
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex);
    ASSERT_TRUE(world.IsValid());

    gobot::PhysicsSceneSnapshot unsupported = MakeShellSnapshot();
    unsupported.deformables[0].model = 2;
    EXPECT_FALSE(world->Build(std::move(unsupported)));
    EXPECT_NE(world->GetLastError().find("model 0"), std::string::npos);

    gobot::PhysicsSceneSnapshot invalid_shell = MakeShellSnapshot();
    invalid_shell.deformables[0].surface_triangles.back() = 99;
    EXPECT_FALSE(world->Build(std::move(invalid_shell)));
    EXPECT_NE(world->GetLastError().find("thin-shell"), std::string::npos);

    gobot::PhysicsSceneSnapshot invalid_tetrahedron = MakeTetrahedronSnapshot();
    invalid_tetrahedron.deformables[0].damping = -1.0;
    EXPECT_FALSE(world->Build(std::move(invalid_tetrahedron)));
    EXPECT_NE(world->GetLastError().find("material parameters"), std::string::npos);
}

TEST(TestSuperDexPhysicsWorld, synchronizes_tetrahedral_vertices_and_nodal_forces) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.0005;
    settings.gravity = gobot::Vector3::Zero();
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeTetrahedronSnapshot())) << world->GetLastError();
    ASSERT_EQ(world->GetSceneState().deformables.size(), 1u);
    EXPECT_EQ(world->GetSceneState().deformables[0].stable_id, 41u);
    ASSERT_EQ(world->GetSceneState().deformables[0].local_vertices.size(), 4u);

    std::vector<gobot::Vector3> forces(4, gobot::Vector3{0.0, 0.0, 0.01});
    ASSERT_TRUE(world->SetDeformableExternalForces(41, forces)) << world->GetLastError();
    world->Step(settings.fixed_time_step);
    ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    for (const gobot::Vector3& vertex :
         world->GetSceneState().deformables[0].local_vertices) {
        EXPECT_TRUE(vertex.allFinite());
    }
    for (const gobot::Vector3& velocity :
         world->GetSceneState().deformables[0].local_velocities) {
        EXPECT_TRUE(velocity.allFinite());
    }
}

TEST(TestSuperDexPhysicsWorld, reports_deformable_contact_forces_only_when_enabled) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.0005;
    settings.gravity = gobot::Vector3::Zero();

    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeTetrahedronSnapshot())) << world->GetLastError();
    world->Step(settings.fixed_time_step);
    ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    EXPECT_TRUE(world->GetSceneState().deformables[0].contact_forces_world.empty());
    world.Reset();

    settings.superdex_solver.record_deformable_contact_forces = true;
    world = server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeTetrahedronSnapshot())) << world->GetLastError();
    world->Step(settings.fixed_time_step);
    ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    const auto& contact_forces =
            world->GetSceneState().deformables[0].contact_forces_world;
    ASSERT_EQ(contact_forces.size(), 4u);
    for (const gobot::Vector3& force : contact_forces) {
        EXPECT_TRUE(force.allFinite());
    }
}

TEST(TestSuperDexPhysicsWorld, reports_volumetric_gravity_in_authored_local_vertices) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.001;
    settings.gravity = gobot::Vector3{0.0, 0.0, -9.81};
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeTetrahedronSnapshot())) << world->GetLastError();

    const auto& initial_vertices =
            world->GetSceneState().deformables[0].local_vertices;
    ASSERT_EQ(initial_vertices.size(), 4u);
    gobot::RealType initial_center_z = 0.0;
    for (const gobot::Vector3& vertex : initial_vertices) {
        initial_center_z += vertex.z();
    }
    initial_center_z /= static_cast<gobot::RealType>(initial_vertices.size());

    for (int step = 0; step < 10; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    }

    const auto& fallen = world->GetSceneState().deformables[0];
    gobot::RealType fallen_center_z = 0.0;
    for (const gobot::Vector3& vertex : fallen.local_vertices) {
        EXPECT_TRUE(vertex.allFinite());
        fallen_center_z += vertex.z();
    }
    fallen_center_z /= static_cast<gobot::RealType>(fallen.local_vertices.size());
    EXPECT_LT(fallen_center_z, initial_center_z - 1.0e-5);
    for (const gobot::Vector3& velocity : fallen.local_velocities) {
        EXPECT_TRUE(velocity.allFinite());
        EXPECT_LT(velocity.z(), 0.0);
    }
}

TEST(TestSuperDexPhysicsWorld, deformable_checkpoint_refreshes_queries_and_replays) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.0005;
    settings.gravity = gobot::Vector3::Zero();
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeTetrahedronSnapshot())) << world->GetLastError();
    const auto initial_vertices =
            world->GetSceneState().deformables[0].local_vertices;
    const std::vector<gobot::Vector3> forces(
            4, gobot::Vector3{0.0, 0.0, 0.01});
    ASSERT_TRUE(world->SetDeformableExternalForces(41, forces)) << world->GetLastError();

    for (int step = 0; step < 4; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    }
    const auto checkpoint_vertices =
            world->GetSceneState().deformables[0].local_vertices;
    const auto checkpoint_velocities =
            world->GetSceneState().deformables[0].local_velocities;
    gobot::Ref<gobot::PhysicsRuntimeCheckpoint> checkpoint =
            world->CaptureCheckpoint();
    ASSERT_TRUE(checkpoint.IsValid());

    for (int step = 0; step < 3; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    }
    const auto replay_vertices =
            world->GetSceneState().deformables[0].local_vertices;

    ASSERT_TRUE(world->RestoreCheckpoint(checkpoint)) << world->GetLastError();
    const auto& restored = world->GetSceneState().deformables[0];
    ASSERT_EQ(restored.local_vertices.size(), checkpoint_vertices.size());
    ASSERT_EQ(restored.local_velocities.size(), checkpoint_velocities.size());
    for (std::size_t index = 0; index < checkpoint_vertices.size(); ++index) {
        EXPECT_TRUE(restored.local_vertices[index].isApprox(
                checkpoint_vertices[index], 1.0e-7));
        EXPECT_TRUE(restored.local_velocities[index].isApprox(
                checkpoint_velocities[index], 1.0e-7));
    }

    for (int step = 0; step < 3; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    }
    const auto& replayed = world->GetSceneState().deformables[0].local_vertices;
    ASSERT_EQ(replayed.size(), replay_vertices.size());
    for (std::size_t index = 0; index < replay_vertices.size(); ++index) {
        EXPECT_TRUE(replayed[index].isApprox(replay_vertices[index], 1.0e-6));
    }

    world->Reset();
    ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    const auto& reset = world->GetSceneState().deformables[0];
    ASSERT_EQ(reset.local_vertices.size(), initial_vertices.size());
    for (std::size_t index = 0; index < initial_vertices.size(); ++index) {
        EXPECT_TRUE(reset.local_vertices[index].isApprox(
                initial_vertices[index], 1.0e-7));
        EXPECT_TRUE(reset.local_velocities[index].isZero(1.0e-7));
    }
}

TEST(TestSuperDexPhysicsWorld, drives_revolute_joint_through_backend_neutral_controller) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.001;
    settings.gravity = gobot::Vector3::Zero();
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeDrivenHingeSnapshot())) << world->GetLastError();
    ASSERT_TRUE(world->SetJointControl(
            "hinge", "hinge_joint", gobot::PhysicsJointControlMode::Position, 0.5));

    for (int step = 0; step < 100; ++step) {
        world->Step(settings.fixed_time_step);
        ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    }
    const gobot::PhysicsJointState& state = world->GetSceneState().robots[0].joints[0];
    EXPECT_GT(state.position, 0.05);
    EXPECT_LT(state.position, 0.55);
    EXPECT_TRUE(std::isfinite(state.velocity));
    EXPECT_TRUE(std::isfinite(state.applied_effort));
    EXPECT_LE(std::abs(state.applied_effort), 20.0 + CMP_EPSILON);
}

TEST(TestSuperDexPhysicsWorld, refreshes_controller_state_between_solver_substeps) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings multi_settings;
    multi_settings.fixed_time_step = 0.002;
    multi_settings.gravity = gobot::Vector3::Zero();
    multi_settings.superdex_solver.substeps = 2;
    gobot::Ref<gobot::PhysicsWorld> multi =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, multi_settings);
    ASSERT_TRUE(multi.IsValid());
    ASSERT_TRUE(multi->Build(MakeDrivenHingeSnapshot())) << multi->GetLastError();

    ASSERT_TRUE(multi->SetJointControl(
            "hinge", "hinge_joint", gobot::PhysicsJointControlMode::Position, 0.5));
    multi->Step(0.002);
    ASSERT_TRUE(multi->GetLastError().empty()) << multi->GetLastError();
    const gobot::PhysicsJointState multi_state =
            multi->GetSceneState().robots[0].joints[0];
    const gobot::PhysicsSolverDiagnostics diagnostics =
            multi->GetSolverDiagnostics();
    EXPECT_GT(diagnostics.total_step_time_seconds, 0.0);
    EXPECT_GE(diagnostics.total_step_time_seconds,
              diagnostics.solve_time_seconds);
    EXPECT_GE(diagnostics.newton_iterations, 0);
    EXPECT_GE(diagnostics.line_search_iterations, 0);
    EXPECT_TRUE(std::isfinite(diagnostics.residual_norm));
    multi.Reset();

    gobot::PhysicsWorldSettings single_settings = multi_settings;
    single_settings.superdex_solver.substeps = 1;
    gobot::Ref<gobot::PhysicsWorld> single =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, single_settings);
    ASSERT_TRUE(single.IsValid());
    ASSERT_TRUE(single->Build(MakeDrivenHingeSnapshot())) << single->GetLastError();
    ASSERT_TRUE(single->SetJointControl(
            "hinge", "hinge_joint", gobot::PhysicsJointControlMode::Position, 0.5));
    single->Step(0.001);
    ASSERT_TRUE(single->GetLastError().empty()) << single->GetLastError();
    const gobot::PhysicsSolverDiagnostics first_substep_diagnostics =
            single->GetSolverDiagnostics();
    single->Step(0.001);
    ASSERT_TRUE(single->GetLastError().empty()) << single->GetLastError();
    const gobot::PhysicsSolverDiagnostics second_substep_diagnostics =
            single->GetSolverDiagnostics();

    const gobot::PhysicsJointState& single_state =
            single->GetSceneState().robots[0].joints[0];
    EXPECT_NEAR(multi_state.position, single_state.position, 1.0e-6);
    EXPECT_NEAR(multi_state.velocity, single_state.velocity, 1.0e-6);
    EXPECT_NEAR(multi_state.applied_effort, single_state.applied_effort, 1.0e-6);
    EXPECT_EQ(diagnostics.newton_iterations,
              first_substep_diagnostics.newton_iterations +
                      second_substep_diagnostics.newton_iterations);
    EXPECT_EQ(diagnostics.line_search_iterations,
              first_substep_diagnostics.line_search_iterations +
                      second_substep_diagnostics.line_search_iterations);
    EXPECT_NEAR(diagnostics.residual_norm,
                std::max(first_substep_diagnostics.residual_norm,
                         second_substep_diagnostics.residual_norm),
                1.0e-7);
}

TEST(TestSuperDexPhysicsWorld, preserves_nonzero_authored_joint_pose_on_build_and_reset) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.001;
    settings.gravity = gobot::Vector3::Zero();
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());

    gobot::PhysicsSceneSnapshot snapshot = MakeDrivenHingeSnapshot();
    constexpr gobot::RealType authored_position = 0.35;
    snapshot.robots[0].joints[0].joint_position = authored_position;
    snapshot.robots[0].joints[0].initial_position = authored_position;
    gobot::Affine3 authored_tip = gobot::Affine3::Identity();
    authored_tip.linear() = gobot::AngleAxis(
            authored_position, gobot::Vector3::UnitZ()).toRotationMatrix();
    authored_tip.translation() =
            authored_tip.linear() * gobot::Vector3{0.15, 0.0, 0.0};
    snapshot.robots[0].links[1].global_transform = authored_tip;
    snapshot.robots[0].links[1].collision_shapes[0].global_transform = authored_tip;

    ASSERT_TRUE(world->Build(std::move(snapshot))) << world->GetLastError();
    ASSERT_EQ(world->GetSceneState().robots.size(), 1u);
    const gobot::PhysicsRobotState& initial = world->GetSceneState().robots[0];
    ASSERT_EQ(initial.joints.size(), 1u);
    ASSERT_EQ(initial.links.size(), 2u);
    EXPECT_NEAR(initial.joints[0].position, authored_position, 1.0e-6);
    EXPECT_TRUE(initial.links[1].global_transform.matrix().isApprox(
            authored_tip.matrix(), 1.0e-6));

    world->Step(settings.fixed_time_step);
    ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    world->Reset();
    ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    const gobot::PhysicsRobotState& reset = world->GetSceneState().robots[0];
    EXPECT_NEAR(reset.joints[0].position, authored_position, 1.0e-6);
    EXPECT_TRUE(reset.links[1].global_transform.matrix().isApprox(
            authored_tip.matrix(), 1.0e-6));
}

TEST(TestSuperDexPhysicsWorld, synchronizes_thin_shell_and_applies_nodal_forces) {
    gobot::PhysicsServer server;
    gobot::PhysicsWorldSettings settings;
    settings.fixed_time_step = 0.0005;
    settings.gravity = gobot::Vector3::Zero();
    gobot::Ref<gobot::PhysicsWorld> world =
            server.CreateWorld(gobot::PhysicsBackendType::SuperDex, settings);
    ASSERT_TRUE(world.IsValid());
    ASSERT_TRUE(world->Build(MakeShellSnapshot())) << world->GetLastError();
    const auto initial_vertices = world->GetSceneState().deformables[0].local_vertices;
    ASSERT_EQ(initial_vertices.size(), 4u);

    const std::vector<gobot::Vector3> forces(4, gobot::Vector3{0.0, 0.0, 0.1});
    ASSERT_TRUE(world->SetDeformableExternalForces(61, forces)) << world->GetLastError();
    world->Step(settings.fixed_time_step);
    ASSERT_TRUE(world->GetLastError().empty()) << world->GetLastError();
    const auto& vertices = world->GetSceneState().deformables[0].local_vertices;
    ASSERT_EQ(vertices.size(), initial_vertices.size());
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        EXPECT_TRUE(vertices[index].allFinite());
        EXPECT_GT(vertices[index].z(), initial_vertices[index].z());
    }
}

#endif
