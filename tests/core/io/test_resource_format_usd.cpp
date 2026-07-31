/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <vector>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/resource_format_usd.hpp>
#include <gobot/physics/physics_scene_compiler.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/convex_mesh_shape_3d.hpp>
#include <gobot/scene/resources/material.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/resources/sphere_shape_3d.hpp>

namespace {

gobot::Node* FindNodeByName(gobot::Node* node, const std::string& name) {
    if (node == nullptr) {
        return nullptr;
    }
    if (node->GetName() == name) {
        return node;
    }
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (gobot::Node* found = FindNodeByName(node->GetChild(static_cast<int>(i)), name)) {
            return found;
        }
    }
    return nullptr;
}

template <typename NodeType>
void CollectNodes(gobot::Node* node, std::vector<NodeType*>* result) {
    if (node == nullptr || result == nullptr) {
        return;
    }
    if (auto* typed = gobot::Object::PointerCastTo<NodeType>(node); typed != nullptr) {
        result->push_back(typed);
    }
    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectNodes(node->GetChild(static_cast<int>(i)), result);
    }
}

} // namespace

TEST(TestResourceFormatUSD, recognizes_usd_extensions_for_packed_scene) {
    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader = gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();

    std::vector<std::string> extensions;
    loader->GetRecognizedExtensionsForType("PackedScene", &extensions);

    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "usd"), extensions.end());
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "usda"), extensions.end());
    EXPECT_NE(std::find(extensions.begin(), extensions.end(), "usdc"), extensions.end());
    EXPECT_TRUE(loader->HandlesType("PackedScene"));
}

TEST(TestResourceFormatUSD, disabled_openusd_loader_fails_without_crashing) {
    if (gobot::ResourceFormatLoaderUSD::IsOpenUSDAvailable()) {
        GTEST_SKIP() << "OpenUSD is enabled in this build.";
    }

    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader = gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();
    gobot::Ref<gobot::Resource> resource = loader->Load("res://missing.usda");
    EXPECT_FALSE(resource.IsValid());
}

TEST(TestResourceFormatUSD, imports_visual_scene_when_openusd_is_available) {
    if (!gobot::ResourceFormatLoaderUSD::IsOpenUSDAvailable()) {
        GTEST_SKIP() << "OpenUSD support is not enabled.";
    }

    gobot::ProjectSettings project_settings;
    const std::filesystem::path fixture =
            std::filesystem::current_path() / "tests/fixtures/usd/visual_scene.usda";
    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader = gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();
    const gobot::Ref<gobot::Resource> resource = loader->Load(fixture.string());
    const gobot::Ref<gobot::PackedScene> packed =
            gobot::dynamic_pointer_cast<gobot::PackedScene>(resource);
    ASSERT_TRUE(packed.IsValid());

    gobot::Node* instance = packed->Instantiate();
    ASSERT_NE(instance, nullptr);
    auto* root = gobot::Object::PointerCastTo<gobot::Node3D>(instance);
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->GetName(), "visual_scene");
    EXPECT_NEAR(root->GetScale().x(), 0.01, 1e-6);
    EXPECT_NEAR(root->GetScale().y(), 0.01, 1e-6);
    EXPECT_NEAR(root->GetScale().z(), 0.01, 1e-6);
    EXPECT_NEAR(root->GetEulerDegree().x(), 90.0, 1e-4);

    auto* world = gobot::Object::PointerCastTo<gobot::Node3D>(FindNodeByName(instance, "World"));
    ASSERT_NE(world, nullptr);
    EXPECT_NEAR(world->GetPosition().x(), 100.0, 1e-5);
    EXPECT_NEAR(world->GetPosition().y(), 200.0, 1e-5);
    EXPECT_NEAR(world->GetPosition().z(), 300.0, 1e-5);
    const gobot::Vector3 world_position =
            (root->GetTransform() * world->GetTransform()).translation();
    EXPECT_NEAR(world_position.x(), 1.0, 1e-5);
    EXPECT_NEAR(world_position.y(), -3.0, 1e-5);
    EXPECT_NEAR(world_position.z(), 2.0, 1e-5);

    auto* quad = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(instance, "Quad"));
    ASSERT_NE(quad, nullptr);
    const gobot::Ref<gobot::ArrayMesh> mesh =
            gobot::dynamic_pointer_cast<gobot::ArrayMesh>(quad->GetMesh());
    ASSERT_TRUE(mesh.IsValid());
    const gobot::MeshSurfaceList surfaces = mesh->GetSurfaces();
    ASSERT_EQ(surfaces.size(), 1);
    EXPECT_EQ(surfaces[0].vertices.size(), 6);
    EXPECT_EQ(surfaces[0].indices.size(), 6);
    ASSERT_EQ(surfaces[0].normals.size(), 6);
    ASSERT_EQ(surfaces[0].uv0.size(), 6);
    for (const gobot::Vector3& normal : surfaces[0].normals) {
        EXPECT_NEAR(normal.x(), 0.0, 1e-6);
        EXPECT_NEAR(normal.y(), 0.0, 1e-6);
        EXPECT_NEAR(normal.z(), 1.0, 1e-6);
    }
    for (const gobot::Vector2 expected : {
                 gobot::Vector2{0.0, 0.0},
                 gobot::Vector2{1.0, 0.0},
                 gobot::Vector2{1.0, 1.0},
                 gobot::Vector2{0.0, 1.0}}) {
        EXPECT_NE(std::find_if(surfaces[0].uv0.begin(), surfaces[0].uv0.end(),
                               [&expected](const gobot::Vector2& actual) {
                                   return actual.isApprox(expected, 1e-6);
                               }),
                  surfaces[0].uv0.end());
    }

    EXPECT_FALSE(surfaces[0].material.IsValid());
    const gobot::Ref<gobot::PBRMaterial3D> material =
            gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(quad->GetMaterial());
    ASSERT_TRUE(material.IsValid());
    EXPECT_NEAR(material->GetAlbedo().red(), 0.2, 1e-6);
    EXPECT_NEAR(material->GetAlbedo().green(), 0.4, 1e-6);
    EXPECT_NEAR(material->GetAlbedo().blue(), 0.8, 1e-6);
    EXPECT_NEAR(material->GetAlbedo().alpha(), 0.75, 1e-6);
    EXPECT_NEAR(material->GetMetallic(), 0.3, 1e-6);
    EXPECT_NEAR(material->GetRoughness(), 0.6, 1e-6);
    EXPECT_EQ(material->GetAlphaMode(), gobot::AlphaMode::Blend);
    EXPECT_TRUE(material->IsDoubleSided());

    auto* hidden = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(instance, "HiddenTriangle"));
    ASSERT_NE(hidden, nullptr);
    EXPECT_FALSE(hidden->IsVisible());
    EXPECT_EQ(FindNodeByName(instance, "BlueMaterial"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "Preview"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "ProxyRepresentation"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "AuxiliaryRoot"), nullptr);

    auto* concave = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(instance, "Concave"));
    ASSERT_NE(concave, nullptr);
    const gobot::Ref<gobot::ArrayMesh> concave_mesh =
            gobot::dynamic_pointer_cast<gobot::ArrayMesh>(concave->GetMesh());
    ASSERT_TRUE(concave_mesh.IsValid());
    const gobot::MeshSurfaceList concave_surfaces = concave_mesh->GetSurfaces();
    ASSERT_EQ(concave_surfaces.size(), 1);
    EXPECT_EQ(concave_surfaces[0].indices.size(), 12);
    double concave_area = 0.0;
    for (std::size_t vertex = 0; vertex < concave_surfaces[0].vertices.size(); vertex += 3) {
        const gobot::Vector3 edge_a = concave_surfaces[0].vertices[vertex + 1] -
                                      concave_surfaces[0].vertices[vertex];
        const gobot::Vector3 edge_b = concave_surfaces[0].vertices[vertex + 2] -
                                      concave_surfaces[0].vertices[vertex];
        concave_area += 0.5 * edge_a.cross(edge_b).norm();
    }
    EXPECT_NEAR(concave_area, 300.0, 1e-5);

    auto* tiny_triangle = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(instance, "TinyTriangle"));
    ASSERT_NE(tiny_triangle, nullptr);
    const gobot::Ref<gobot::ArrayMesh> tiny_mesh =
            gobot::dynamic_pointer_cast<gobot::ArrayMesh>(tiny_triangle->GetMesh());
    ASSERT_TRUE(tiny_mesh.IsValid());
    const gobot::MeshSurfaceList tiny_surfaces = tiny_mesh->GetSurfaces();
    ASSERT_EQ(tiny_surfaces.size(), 1);
    EXPECT_EQ(tiny_surfaces[0].indices.size(), 3);

    auto* rotated = gobot::Object::PointerCastTo<gobot::Node3D>(
            FindNodeByName(instance, "Rotated"));
    ASSERT_NE(rotated, nullptr);
    const gobot::Matrix3 expected_rotation = (gobot::Matrix3() <<
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0).finished();
    EXPECT_TRUE(rotated->GetTransform().linear().isApprox(expected_rotation, 1e-5));

    gobot::Object::Delete(instance);
}

TEST(TestResourceFormatUSD, composes_payloads_and_references_without_physics_metadata_nodes) {
    if (!gobot::ResourceFormatLoaderUSD::IsOpenUSDAvailable()) {
        GTEST_SKIP() << "OpenUSD support is not enabled.";
    }

    gobot::ProjectSettings project_settings;
    const std::filesystem::path fixture =
            std::filesystem::current_path() / "tests/fixtures/usd/structured_scene.usda";
    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader = gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();
    const gobot::Ref<gobot::PackedScene> packed = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            loader->Load(fixture.string()));
    ASSERT_TRUE(packed.IsValid());

    gobot::Node* instance = packed->Instantiate();
    ASSERT_NE(instance, nullptr);
    auto* robot = gobot::Object::PointerCastTo<gobot::Node3D>(FindNodeByName(instance, "Robot"));
    auto* base_link = gobot::Object::PointerCastTo<gobot::Node3D>(
            FindNodeByName(instance, "base_link"));
    auto* visual = gobot::Object::PointerCastTo<gobot::Node3D>(FindNodeByName(instance, "Visual"));
    auto* surface = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(instance, "Surface"));
    ASSERT_NE(robot, nullptr);
    ASSERT_NE(base_link, nullptr);
    ASSERT_NE(visual, nullptr);
    ASSERT_NE(surface, nullptr);

    EXPECT_EQ(robot->GetParent(), instance);
    EXPECT_EQ(base_link->GetParent(), robot);
    EXPECT_EQ(visual->GetParent(), base_link);
    EXPECT_EQ(surface->GetParent(), visual);
    EXPECT_TRUE(robot->GetPosition().isApprox(gobot::Vector3{1.0, 2.0, 3.0}, 1e-6));
    EXPECT_TRUE(base_link->GetPosition().isApprox(gobot::Vector3{0.0, 0.0, 0.5}, 1e-6));
    EXPECT_TRUE(visual->GetPosition().isApprox(gobot::Vector3{0.0, 0.0, 2.0}, 1e-6));

    const gobot::Ref<gobot::ArrayMesh> mesh =
            gobot::dynamic_pointer_cast<gobot::ArrayMesh>(surface->GetMesh());
    ASSERT_TRUE(mesh.IsValid());
    const gobot::MeshSurfaceList surfaces = mesh->GetSurfaces();
    ASSERT_EQ(surfaces.size(), 1);
    EXPECT_EQ(surfaces[0].vertices.size(), 3);

    EXPECT_EQ(FindNodeByName(instance, "PhysicsOnly"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "Scene"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "KneeJoint"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "Actuators"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "KneeDrive"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "PayloadMetadata"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "ReferenceMetadata"), nullptr);

    gobot::Object::Delete(instance);
}

TEST(TestResourceFormatUSD, imports_articulation_nodes_and_unique_collision_names) {
    if (!gobot::ResourceFormatLoaderUSD::IsOpenUSDAvailable()) {
        GTEST_SKIP() << "OpenUSD support is not enabled.";
    }

    gobot::ProjectSettings project_settings;
    const std::filesystem::path fixture =
            std::filesystem::current_path() / "tests/fixtures/usd/physics_articulation.usda";
    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader = gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();
    const gobot::Ref<gobot::PackedScene> packed = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            loader->Load(fixture.string()));
    ASSERT_TRUE(packed.IsValid());

    gobot::Node* instance = packed->Instantiate();
    ASSERT_NE(instance, nullptr);

    std::vector<gobot::Robot3D*> robots;
    std::vector<gobot::Link3D*> links;
    std::vector<gobot::Joint3D*> joints;
    std::vector<gobot::CollisionShape3D*> collisions;
    std::vector<gobot::MeshInstance3D*> visuals;
    CollectNodes(instance, &robots);
    CollectNodes(instance, &links);
    CollectNodes(instance, &joints);
    CollectNodes(instance, &collisions);
    CollectNodes(instance, &visuals);
    ASSERT_EQ(robots.size(), 1);
    ASSERT_EQ(links.size(), 2);
    ASSERT_EQ(joints.size(), 2);
    ASSERT_EQ(collisions.size(), 4);
    ASSERT_EQ(visuals.size(), 2);

    gobot::Robot3D* robot = robots.front();
    EXPECT_EQ(robot->GetName(), "TinyRobot");
    EXPECT_EQ(robot->GetSourcePath(), fixture.string());
    EXPECT_TRUE(robot->GetPosition().isApprox(gobot::Vector3{1.0, 2.0, 3.0}, 1e-6));

    auto* floating = gobot::Object::PointerCastTo<gobot::Joint3D>(
            FindNodeByName(instance, "floating_base_joint"));
    auto* base = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(instance, "base"));
    auto* shoulder = gobot::Object::PointerCastTo<gobot::Joint3D>(
            FindNodeByName(instance, "shoulder"));
    auto* arm = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(instance, "arm"));
    ASSERT_NE(floating, nullptr);
    ASSERT_NE(base, nullptr);
    ASSERT_NE(shoulder, nullptr);
    ASSERT_NE(arm, nullptr);
    EXPECT_EQ(floating->GetParent(), robot);
    EXPECT_EQ(base->GetParent(), floating);
    EXPECT_EQ(shoulder->GetParent(), base);
    EXPECT_EQ(arm->GetParent(), shoulder);

    EXPECT_EQ(floating->GetJointType(), gobot::JointType::Floating);
    EXPECT_EQ(floating->GetChildLink(), "base");
    EXPECT_TRUE(floating->GetParentLink().empty());
    EXPECT_TRUE(base->HasInertial());
    EXPECT_NEAR(base->GetMass(), 2.5, 1e-6);
    EXPECT_TRUE(base->GetCenterOfMass().isApprox(gobot::Vector3{0.1, 0.2, 0.3}, 1e-6));
    EXPECT_TRUE(base->GetInertiaDiagonal().isApprox(gobot::Vector3{0.4, 0.5, 0.6}, 1e-6));
    EXPECT_TRUE(arm->HasInertial());
    EXPECT_NEAR(arm->GetMass(), 1.25, 1e-6);

    EXPECT_EQ(shoulder->GetJointType(), gobot::JointType::Revolute);
    EXPECT_EQ(shoulder->GetParentLink(), "base");
    EXPECT_EQ(shoulder->GetChildLink(), "arm");
    EXPECT_TRUE(shoulder->GetAxis().isApprox(gobot::Vector3::UnitZ(), 1e-6));
    EXPECT_NEAR(shoulder->GetLowerLimit(), -gobot::DEG_TO_RAD(90.0), 1e-6);
    EXPECT_NEAR(shoulder->GetUpperLimit(), gobot::DEG_TO_RAD(45.0), 1e-6);
    EXPECT_EQ(shoulder->GetDriveMode(), gobot::JointDriveMode::Position);
    EXPECT_NEAR(shoulder->GetDriveStiffness(), 12.0, 1e-6);
    EXPECT_NEAR(shoulder->GetDriveDamping(), 3.0, 1e-6);
    EXPECT_NEAR(shoulder->GetEffortLimit(), 20.0, 1e-6);
    EXPECT_NEAR(shoulder->GetInitialPosition(), gobot::DEG_TO_RAD(15.0), 1e-6);
    EXPECT_TRUE(shoulder->IsAffineActuatorEnabled());
    EXPECT_NEAR(shoulder->GetAffineActuatorControlGain(), 18.0, 1e-6);
    EXPECT_NEAR(shoulder->GetAffineActuatorForceOffset(), 0.25, 1e-6);
    EXPECT_NEAR(shoulder->GetAffineActuatorPositionGain(), -18.0, 1e-6);
    EXPECT_NEAR(shoulder->GetAffineActuatorVelocityGain(), 0.75, 1e-6);
    EXPECT_NEAR(shoulder->GetAffineActuatorInheritRange(), 0.8, 1e-6);
    EXPECT_TRUE(arm->GetPosition().isApprox(gobot::Vector3{0.0, 0.0, 0.5}, 1e-6));

    auto* base_collision = gobot::Object::PointerCastTo<gobot::CollisionShape3D>(
            FindNodeByName(instance, "base_collider_collision"));
    auto* shared_visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(instance, "shared_collider"));
    auto* shared_collision = gobot::Object::PointerCastTo<gobot::CollisionShape3D>(
            FindNodeByName(instance, "base_shared_collider_collision"));
    auto* arm_collision = gobot::Object::PointerCastTo<gobot::CollisionShape3D>(
            FindNodeByName(instance, "arm_collider_collision"));
    auto* arm_self_collision = gobot::Object::PointerCastTo<gobot::CollisionShape3D>(
            FindNodeByName(instance, "arm_arm_collision"));
    const auto arm_visual = std::find_if(
            visuals.begin(), visuals.end(),
            [](const gobot::MeshInstance3D* visual) { return visual->GetName() == "arm"; });
    ASSERT_NE(base_collision, nullptr);
    ASSERT_NE(shared_visual, nullptr);
    ASSERT_NE(shared_collision, nullptr);
    ASSERT_NE(arm_collision, nullptr);
    ASSERT_NE(arm_self_collision, nullptr);
    ASSERT_NE(arm_visual, visuals.end());
    EXPECT_EQ(base_collision->GetParent(), base);
    EXPECT_EQ(shared_visual->GetParent(), base);
    EXPECT_EQ(shared_collision->GetParent(), base);
    EXPECT_EQ(arm_collision->GetParent(), arm);
    EXPECT_EQ(arm_self_collision->GetParent(), arm);
    EXPECT_EQ((*arm_visual)->GetParent(), arm);
    EXPECT_FALSE((*arm_visual)->IsVisible());
    EXPECT_NE(base_collision->GetName(), arm_collision->GetName());
    EXPECT_EQ(FindNodeByName(instance, "collider"), nullptr);
    EXPECT_FALSE(base_collision->IsVisible());
    EXPECT_FALSE(arm_collision->IsVisible());
    EXPECT_EQ(base_collision->GetContactType(), 1);
    EXPECT_EQ(base_collision->GetContactAffinity(), 1);
    EXPECT_EQ(shared_collision->GetContactType(), 2);
    EXPECT_EQ(shared_collision->GetContactAffinity(), 4);
    EXPECT_TRUE(base_collision->GetPosition().isApprox(gobot::Vector3{0.1, 0.0, 0.0}, 1e-6));

    const gobot::Ref<gobot::SphereShape3D> sphere =
            gobot::dynamic_pointer_cast<gobot::SphereShape3D>(base_collision->GetShape());
    const gobot::Ref<gobot::BoxShape3D> box =
            gobot::dynamic_pointer_cast<gobot::BoxShape3D>(arm_collision->GetShape());
    const gobot::Ref<gobot::ConvexMeshShape3D> convex =
            gobot::dynamic_pointer_cast<gobot::ConvexMeshShape3D>(shared_collision->GetShape());
    ASSERT_TRUE(sphere.IsValid());
    ASSERT_TRUE(box.IsValid());
    ASSERT_TRUE(convex.IsValid());
    ASSERT_TRUE(convex->GetMesh().IsValid());
    EXPECT_NEAR(sphere->GetRadius(), 0.2, 1e-6);
    EXPECT_TRUE(box->GetSize().isApprox(gobot::Vector3{0.4, 0.8, 1.2}, 1e-6));

    gobot::CompiledPhysicsScene compiled;
    std::string compile_error;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(instance, &compiled, &compile_error))
            << compile_error;
    ASSERT_EQ(compiled.snapshot.robots.size(), 1);
    std::size_t collision_count = 0;
    for (const gobot::PhysicsLinkSnapshot& link : compiled.snapshot.robots.front().links) {
        collision_count += link.collision_shapes.size();
    }
    EXPECT_EQ(collision_count, 4);
    ASSERT_EQ(compiled.snapshot.robots.front().joints.size(), 2);
    const auto imported_joint = std::find_if(
            compiled.snapshot.robots.front().joints.begin(),
            compiled.snapshot.robots.front().joints.end(),
            [](const gobot::PhysicsJointSnapshot& value) { return value.name == "shoulder"; });
    ASSERT_NE(imported_joint, compiled.snapshot.robots.front().joints.end());
    EXPECT_TRUE(imported_joint->affine_actuator_enabled);
    EXPECT_DOUBLE_EQ(imported_joint->affine_actuator_control_gain, 18.0);
    EXPECT_DOUBLE_EQ(imported_joint->affine_actuator_position_gain, -18.0);

    gobot::Object::Delete(instance);
}

TEST(TestResourceFormatUSD, imports_world_fixed_articulation_without_floating_base) {
    if (!gobot::ResourceFormatLoaderUSD::IsOpenUSDAvailable()) {
        GTEST_SKIP() << "OpenUSD support is not enabled.";
    }

    gobot::ProjectSettings project_settings;
    const std::filesystem::path fixture =
            std::filesystem::current_path() / "tests/fixtures/usd/fixed_articulation.usda";
    gobot::Ref<gobot::ResourceFormatLoaderUSD> loader =
            gobot::MakeRef<gobot::ResourceFormatLoaderUSD>();
    const gobot::Ref<gobot::PackedScene> packed = gobot::dynamic_pointer_cast<gobot::PackedScene>(
            loader->Load(fixture.string()));
    ASSERT_TRUE(packed.IsValid());

    gobot::Node* instance = packed->Instantiate();
    ASSERT_NE(instance, nullptr);
    auto* robot = gobot::Object::PointerCastTo<gobot::Robot3D>(
            FindNodeByName(instance, "FixedRobot"));
    auto* base = gobot::Object::PointerCastTo<gobot::Link3D>(FindNodeByName(instance, "base"));
    ASSERT_NE(robot, nullptr);
    ASSERT_NE(base, nullptr);
    EXPECT_EQ(base->GetParent(), robot);
    EXPECT_TRUE(base->HasInertial());
    EXPECT_DOUBLE_EQ(base->GetMass(), 1.0);
    EXPECT_TRUE(base->GetCenterOfMass().isApprox(gobot::Vector3::Zero(), 1e-9));
    EXPECT_TRUE(base->GetInertiaDiagonal().isApprox(gobot::Vector3::Zero(), 1e-9));
    EXPECT_TRUE(base->GetInertiaOrientation().isApprox(gobot::Quaternion::Identity(), 1e-9));
    EXPECT_EQ(FindNodeByName(instance, "floating_base_joint"), nullptr);
    EXPECT_EQ(FindNodeByName(instance, "world_anchor"), nullptr);

    std::vector<gobot::Joint3D*> joints;
    CollectNodes(instance, &joints);
    EXPECT_TRUE(joints.empty());

    gobot::CompiledPhysicsScene compiled;
    std::string compile_error;
    ASSERT_TRUE(gobot::PhysicsSceneCompiler::Compile(instance, &compiled, &compile_error))
            << compile_error;
    ASSERT_EQ(compiled.snapshot.robots.size(), 1);
    ASSERT_EQ(compiled.snapshot.robots.front().links.size(), 1);
    EXPECT_TRUE(compiled.snapshot.robots.front().joints.empty());

    gobot::Object::Delete(instance);
}
