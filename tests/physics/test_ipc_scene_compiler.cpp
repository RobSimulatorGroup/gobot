/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <gobot/core/sha256.hpp>
#include <gobot/physics/ipc_scene_compiler.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/deformable_body_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/convex_mesh_shape_3d.hpp>
#include <gobot/scene/resources/tetrahedral_mesh.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/tactile_sensor_3d.hpp>
#include <gobot/scene/window.hpp>

namespace {

gobot::Ref<gobot::TetrahedralMesh> MakeTetrahedron() {
    auto mesh = gobot::MakeRef<gobot::TetrahedralMesh>();
    mesh->SetVertices({
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
    });
    mesh->SetTetrahedra({0, 1, 2, 3});
    return mesh;
}

} // namespace

TEST(TestIpcSceneCompiler, sha256_matches_standard_vectors) {
    EXPECT_EQ(gobot::Sha256Hex(""),
              "e3b0c44298fc1c149afbf4c8996fb924"
              "27ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(gobot::Sha256Hex("abc"),
              "ba7816bf8f01cfea414140de5dae2223"
              "b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(gobot::Sha256Hex(
                      "abcdbcdecdefdefgefghfghighijhijk"
                      "ijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039"
              "a33ce45964ff2167f6ecedd419db06c1");
}

TEST(TestIpcSceneCompiler, compiles_deterministic_content_addressed_artifact) {
    gobot::SceneTree tree(false);
    tree.Initialize();
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("ipc_world");
    tree.GetRoot()->AddChild(root);

    const gobot::Ref<gobot::TetrahedralMesh> mesh = MakeTetrahedron();
    auto* body = gobot::Object::New<gobot::DeformableBody3D>();
    body->SetName("soft_body");
    body->SetMesh(mesh);
    body->SetDensity(1050.0);
    body->SetSelfCollisionEnabled(true);
    root->AddChild(body);

    auto config = gobot::MakeRef<gobot::TactileSensorConfig>();
    config->SetImageWidth(16);
    config->SetImageHeight(12);
    config->SetCoatVertexIndices({0, 1, 2});
    config->SetStickVertexIndices({3});
    config->SetMarkerPositions({{4.0, 5.0}});
    config->SetMarkerTetrahedra({0});
    config->SetMarkerBarycentric({{0.25, 0.25, 0.25, 0.25}});
    auto* sensor = gobot::Object::New<gobot::TactileSensor3D>();
    sensor->SetName("finger_pad");
    sensor->SetGelMesh(mesh);
    sensor->SetConfig(config);
    root->AddChild(sensor);

    gobot::IpcSceneArtifact first;
    gobot::IpcSceneArtifact second;
    std::string error;
    ASSERT_TRUE(gobot::IpcSceneCompiler::Compile(root, &first, &error)) << error;
    ASSERT_TRUE(gobot::IpcSceneCompiler::Compile(root, &second, &error)) << error;

    EXPECT_EQ(first.schema_version, 1);
    EXPECT_EQ(first.producer, "gobot");
    EXPECT_EQ(first.format, "gobot-ipc");
    EXPECT_EQ(first.manifest, second.manifest);
    EXPECT_EQ(first.manifest_sha256, second.manifest_sha256);
    EXPECT_EQ(first.manifest_sha256, gobot::Sha256Digest(first.manifest));
    ASSERT_EQ(first.blobs.size(), 1);
    EXPECT_EQ(first.blobs[0].id, first.blobs[0].sha256);
    EXPECT_EQ(first.blobs[0].sha256,
              gobot::Sha256Digest(std::span<const std::uint8_t>(first.blobs[0].data)));

    const nlohmann::json manifest = nlohmann::json::parse(first.manifest);
    EXPECT_EQ(manifest.at("scene_name"), "ipc_world");
    ASSERT_EQ(manifest.at("deformable_bodies").size(), 1);
    ASSERT_EQ(manifest.at("tactile_sensors").size(), 1);
    ASSERT_EQ(manifest.at("blobs").size(), 1);
    EXPECT_EQ(manifest.at("deformable_bodies").at(0).at("mesh_blob"),
              manifest.at("tactile_sensors").at(0).at("gel_mesh_blob"));
    EXPECT_EQ(manifest.at("tactile_sensors").at(0).at("resolution"),
              nlohmann::json::array({12, 16}));
    EXPECT_EQ(manifest.at("tactile_sensors").at(0).at("coat_vertex_indices"),
              nlohmann::json::array({0, 1, 2}));

    tree.Finalize();
}

TEST(TestIpcSceneCompiler, rejects_invalid_deformable_mesh) {
    gobot::SceneTree tree(false);
    tree.Initialize();
    auto* body = gobot::Object::New<gobot::DeformableBody3D>();
    body->SetName("invalid");
    auto mesh = MakeTetrahedron();
    mesh->SetTetrahedra({0, 2, 1, 3});
    body->SetMesh(mesh);
    tree.GetRoot()->AddChild(body);

    gobot::IpcSceneArtifact artifact;
    std::string error;
    EXPECT_FALSE(gobot::IpcSceneCompiler::Compile(body, &artifact, &error));
    EXPECT_NE(error.find("positively oriented"), std::string::npos);
    tree.Finalize();
}

TEST(TestIpcSceneCompiler, accepts_small_uniform_scale_and_rejects_invalid_surface) {
    gobot::SceneTree tree(false);
    tree.Initialize();
    auto* body = gobot::Object::New<gobot::DeformableBody3D>();
    body->SetName("small_body");
    body->SetScale(gobot::Vector3::Constant(1.0e-4));
    body->SetMesh(MakeTetrahedron());
    tree.GetRoot()->AddChild(body);

    gobot::IpcSceneArtifact artifact;
    std::string error;
    ASSERT_TRUE(gobot::IpcSceneCompiler::Compile(body, &artifact, &error)) << error;

    auto invalid_surface = MakeTetrahedron();
    invalid_surface->SetSurfaceTriangles({0, 1, 2});
    body->SetMesh(invalid_surface);
    EXPECT_FALSE(gobot::IpcSceneCompiler::Compile(body, &artifact, &error));
    EXPECT_NE(error.find("every boundary face"), std::string::npos);

    auto isolated_vertex = MakeTetrahedron();
    auto vertices = isolated_vertex->GetVertices();
    vertices.push_back({2.0, 2.0, 2.0});
    isolated_vertex->SetVertices(vertices);
    body->SetMesh(isolated_vertex);
    EXPECT_FALSE(gobot::IpcSceneCompiler::Compile(body, &artifact, &error));
    EXPECT_NE(error.find("not referenced"), std::string::npos);

    auto overlapping_tetrahedra = MakeTetrahedron();
    auto overlapping_vertices = overlapping_tetrahedra->GetVertices();
    overlapping_vertices.push_back({0.0, 0.0, 2.0});
    overlapping_tetrahedra->SetVertices(overlapping_vertices);
    overlapping_tetrahedra->SetTetrahedra({0, 1, 2, 3, 0, 1, 2, 4});
    body->SetMesh(overlapping_tetrahedra);
    EXPECT_FALSE(gobot::IpcSceneCompiler::Compile(body, &artifact, &error));
    EXPECT_NE(error.find("opposite orientation"), std::string::npos);

    auto duplicate_tetrahedra = MakeTetrahedron();
    duplicate_tetrahedra->SetTetrahedra({0, 1, 2, 3, 0, 1, 2, 3});
    body->SetMesh(duplicate_tetrahedra);
    EXPECT_FALSE(gobot::IpcSceneCompiler::Compile(body, &artifact, &error));
    EXPECT_NE(error.find("duplicate tetrahedron"), std::string::npos);
    tree.Finalize();
}

TEST(TestIpcSceneCompiler, records_robot_fk_and_inertial_topology) {
    gobot::SceneTree tree(false);
    tree.Initialize();
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("hand");
    tree.GetRoot()->AddChild(robot);

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    base->SetHasInertial(true);
    base->SetMass(1.5);
    base->SetCenterOfMass({0.1, 0.2, 0.3});
    base->SetInertiaDiagonal({1.0, 2.0, 3.0});
    robot->AddChild(base);

    auto box = gobot::MakeRef<gobot::BoxShape3D>();
    box->SetSize({0.2, 0.3, 0.4});
    auto* base_collision = gobot::Object::New<gobot::CollisionShape3D>();
    base_collision->SetName("base_collision");
    base_collision->SetShape(gobot::dynamic_pointer_cast<gobot::Shape3D>(box));
    base->AddChild(base_collision);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("finger_joint");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("finger");
    joint->SetAxis({0.0, 1.0, 0.0});
    joint->SetLowerLimit(-0.5);
    joint->SetUpperLimit(0.8);
    joint->SetEffortLimit(2.0);
    joint->SetVelocityLimit(3.0);
    base->AddChild(joint);

    auto* finger = gobot::Object::New<gobot::Link3D>();
    finger->SetName("finger");
    joint->AddChild(finger);

    auto triangle_mesh = gobot::MakeRef<gobot::ArrayMesh>();
    triangle_mesh->SetSurface(
            {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.0, 0.1, 0.0}},
            {0, 1, 2});
    auto convex = gobot::MakeRef<gobot::ConvexMeshShape3D>();
    convex->SetMesh(gobot::dynamic_pointer_cast<gobot::Mesh>(triangle_mesh));
    auto* finger_collision = gobot::Object::New<gobot::CollisionShape3D>();
    finger_collision->SetName("finger_collision");
    finger_collision->SetShape(gobot::dynamic_pointer_cast<gobot::Shape3D>(convex));
    finger->AddChild(finger_collision);

    auto tactile_config = gobot::MakeRef<gobot::TactileSensorConfig>();
    auto* tactile = gobot::Object::New<gobot::TactileSensor3D>();
    tactile->SetName("finger_tactile");
    tactile->SetPosition({0.0, 0.0, 0.02});
    tactile->SetGelMesh(MakeTetrahedron());
    tactile->SetConfig(tactile_config);
    finger->AddChild(tactile);

    gobot::IpcSceneArtifact artifact;
    std::string error;
    ASSERT_TRUE(gobot::IpcSceneCompiler::Compile(robot, &artifact, &error)) << error;
    const nlohmann::json manifest = nlohmann::json::parse(artifact.manifest);
    ASSERT_EQ(manifest.at("robots").size(), 1);
    const nlohmann::json& compiled_robot = manifest.at("robots").at(0);
    ASSERT_EQ(compiled_robot.at("links").size(), 2);
    ASSERT_EQ(compiled_robot.at("joints").size(), 1);
    EXPECT_EQ(compiled_robot.at("root_link_paths").size(), 1);
    EXPECT_EQ(compiled_robot.at("links").at(0).at("mass"), 1.5);
    const nlohmann::json& base_shapes =
            compiled_robot.at("links").at(0).at("collision_shapes");
    ASSERT_EQ(base_shapes.size(), 1);
    EXPECT_EQ(base_shapes.at(0).at("shape_type"), "box");
    EXPECT_NEAR(base_shapes.at(0).at("size").at(0), 0.2, 1.0e-7);
    EXPECT_NEAR(base_shapes.at(0).at("size").at(1), 0.3, 1.0e-7);
    EXPECT_NEAR(base_shapes.at(0).at("size").at(2), 0.4, 1.0e-7);
    const nlohmann::json& finger_shapes =
            compiled_robot.at("links").at(1).at("collision_shapes");
    ASSERT_EQ(finger_shapes.size(), 1);
    EXPECT_EQ(finger_shapes.at(0).at("shape_type"), "triangle_mesh");
    EXPECT_EQ(finger_shapes.at(0).at("vertex_count"), 3);
    EXPECT_EQ(finger_shapes.at(0).at("triangle_count"), 1);
    ASSERT_EQ(manifest.at("tactile_sensors").size(), 1);
    const nlohmann::json& attachment =
            manifest.at("tactile_sensors").at(0).at("attachment");
    EXPECT_EQ(attachment.at("link_path"),
              compiled_robot.at("links").at(1).at("path"));
    EXPECT_NEAR(attachment.at("transform").at("matrix_row_major").at(11),
                0.02, 1.0e-8);
    ASSERT_EQ(manifest.at("blobs").size(), 2);
    EXPECT_EQ(manifest.at("blobs").at(0).at("id") <
                      manifest.at("blobs").at(1).at("id"),
              true);
    const auto triangle_blob = std::find_if(
            manifest.at("blobs").begin(), manifest.at("blobs").end(),
            [](const nlohmann::json& blob) {
                return blob.at("encoding") == "gobot.triangle-mesh.le.v1";
            });
    ASSERT_NE(triangle_blob, manifest.at("blobs").end());
    EXPECT_EQ(triangle_blob->at("id"), finger_shapes.at(0).at("mesh_blob"));
    const nlohmann::json& compiled_joint = compiled_robot.at("joints").at(0);
    EXPECT_EQ(compiled_joint.at("parent_link"), "base");
    EXPECT_EQ(compiled_joint.at("child_link"), "finger");
    EXPECT_TRUE(compiled_joint.at("parent_link_path").get<std::string>().ends_with("/base"));
    EXPECT_TRUE(compiled_joint.at("child_link_path").get<std::string>().ends_with("/finger"));
    EXPECT_EQ(compiled_joint.at("axis"), nlohmann::json::array({0.0, 1.0, 0.0}));
    EXPECT_EQ(compiled_joint.at("local_transform").at("matrix_row_major").size(), 16);
    tree.Finalize();
}
