/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/resource_format_usd.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/material.hpp>
#include <gobot/scene/resources/packed_scene.hpp>

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

    const gobot::Ref<gobot::PBRMaterial3D> material =
            gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(surfaces[0].material);
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
