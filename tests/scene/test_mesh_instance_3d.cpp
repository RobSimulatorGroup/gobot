/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <gobot/rendering/render_server.hpp>
#include <gobot/rendering/scene_render_items.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/terrain_3d.hpp>
#include <gobot/scene/window.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>

TEST(TestMeshInstance3D, stores_mesh_resource_and_transform) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    auto* mesh_instance = gobot::Object::New<gobot::MeshInstance3D>();
    gobot::Ref<gobot::BoxMesh> box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetSize({2.0f, 3.0f, 4.0f});

    mesh_instance->SetMesh(box_mesh);
    mesh_instance->SetPosition({1.0f, 2.0f, 3.0f});

    gobot::Ref<gobot::BoxMesh> stored_box = gobot::dynamic_pointer_cast<gobot::BoxMesh>(mesh_instance->GetMesh());
    ASSERT_TRUE(stored_box.IsValid());
    EXPECT_TRUE(stored_box->GetSize().isApprox(gobot::Vector3(2.0f, 3.0f, 4.0f), CMP_EPSILON));
    EXPECT_TRUE(mesh_instance->GetPosition().isApprox(gobot::Vector3(1.0f, 2.0f, 3.0f), CMP_EPSILON));

    gobot::Object::Delete(mesh_instance);
}

TEST(TestMeshInstance3D, is_visible_by_default_when_inside_tree) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    auto* mesh_instance = gobot::Object::New<gobot::MeshInstance3D>();
    mesh_instance->SetMesh(gobot::MakeRef<gobot::BoxMesh>());
    tree->GetRoot()->AddChild(mesh_instance, true);

    EXPECT_TRUE(mesh_instance->IsVisible());
    EXPECT_TRUE(mesh_instance->IsVisibleInTree());

    gobot::Object::Delete(tree);
}

TEST(TestMeshInstance3D, hidden_parent_excludes_visible_child_mesh_from_render_items) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    auto* parent = gobot::Object::New<gobot::Node3D>();
    auto* mesh_instance = gobot::Object::New<gobot::MeshInstance3D>();
    mesh_instance->SetMesh(gobot::MakeRef<gobot::BoxMesh>());
    parent->AddChild(mesh_instance, true);
    tree->GetRoot()->AddChild(parent, true);

    gobot::SceneRenderItems items = gobot::CollectSceneRenderItems(tree->GetRoot());
    EXPECT_EQ(items.visual_meshes.size(), 1);

    parent->SetVisible(false);
    EXPECT_TRUE(mesh_instance->IsVisible());
    EXPECT_FALSE(mesh_instance->IsVisibleInTree());

    items = gobot::CollectSceneRenderItems(tree->GetRoot());
    EXPECT_TRUE(items.visual_meshes.empty());

    parent->SetVisible(true);
    mesh_instance->SetVisible(false);
    items = gobot::CollectSceneRenderItems(tree->GetRoot());
    EXPECT_TRUE(items.visual_meshes.empty());

    gobot::Object::Delete(tree);
}

TEST(TestMeshInstance3D, hidden_collision_shapes_are_excluded_from_debug_rendering) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    auto* collision_shape = gobot::Object::New<gobot::CollisionShape3D>();
    collision_shape->SetShape(gobot::MakeRef<gobot::BoxShape3D>());
    collision_shape->SetVisible(false);
    tree->GetRoot()->AddChild(collision_shape, true);

    const gobot::SceneRenderItems items = gobot::CollectSceneRenderItems(tree->GetRoot());
    EXPECT_TRUE(items.collision_shapes.empty());

    gobot::Object::Delete(tree);
}

TEST(TestMeshInstance3D, terrain_vertex_colors_are_not_tinted_by_surface_color) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    auto* terrain = gobot::Object::New<gobot::Terrain3D>();
    terrain->SetSurfaceColor({0.2f, 0.3f, 0.4f, 0.75f});
    terrain->AddBox({0.0, 0.0, -0.5}, {1.0, 1.0, 1.0});
    terrain->SetColorMode(gobot::TerrainColorMode::Palette);
    tree->GetRoot()->AddChild(terrain, true);

    gobot::SceneRenderItems items = gobot::CollectSceneRenderItems(tree->GetRoot());
    ASSERT_EQ(items.visual_meshes.size(), 1);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].surface_color.red(), 1.0f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].surface_color.green(), 1.0f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].surface_color.blue(), 1.0f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].surface_color.alpha(), 0.75f);

    terrain->SetColorMode(gobot::TerrainColorMode::SurfaceColor);
    items = gobot::CollectSceneRenderItems(tree->GetRoot());
    ASSERT_EQ(items.visual_meshes.size(), 1);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].surface_color.red(), 0.2f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].surface_color.green(), 0.3f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].surface_color.blue(), 0.4f);

    gobot::Object::Delete(tree);
}

TEST(TestMeshInstance3D, node3d_transform_properties_are_reflected) {
    auto position = gobot::Type::get<gobot::Node3D>().get_property("position");
    auto rotation = gobot::Type::get<gobot::Node3D>().get_property("rotation_degrees");
    auto scale = gobot::Type::get<gobot::Node3D>().get_property("scale");
    auto mesh = gobot::Type::get<gobot::MeshInstance3D>().get_property("mesh");
    auto material = gobot::Type::get<gobot::MeshInstance3D>().get_property("material");
    auto material_override = gobot::Type::get<gobot::MeshInstance3D>().get_property("material_override");

    EXPECT_TRUE(position.is_valid());
    EXPECT_TRUE(rotation.is_valid());
    EXPECT_TRUE(scale.is_valid());
    EXPECT_TRUE(mesh.is_valid());
    EXPECT_TRUE(material.is_valid());
    EXPECT_FALSE(material_override.is_valid());
}

TEST(TestMeshInstance3D, active_material_prefers_instance_material_over_mesh_material) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    auto* mesh_instance = gobot::Object::New<gobot::MeshInstance3D>();
    gobot::Ref<gobot::BoxMesh> mesh = gobot::MakeRef<gobot::BoxMesh>();
    gobot::Ref<gobot::PBRMaterial3D> mesh_material = gobot::MakeRef<gobot::PBRMaterial3D>();
    gobot::Ref<gobot::PBRMaterial3D> instance_material = gobot::MakeRef<gobot::PBRMaterial3D>();

    mesh->SetMaterial(mesh_material);
    mesh_instance->SetMesh(mesh);

    EXPECT_EQ(mesh_instance->GetActiveMaterial().Get(), mesh_material.Get());

    mesh_instance->SetMaterial(instance_material);
    EXPECT_EQ(mesh_instance->GetActiveMaterial().Get(), instance_material.Get());

    mesh_instance->SetMaterial({});
    EXPECT_EQ(mesh_instance->GetActiveMaterial().Get(), mesh_material.Get());

    gobot::Object::Delete(mesh_instance);
}
