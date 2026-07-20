/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <gobot/rendering/render_server.hpp>
#include <gobot/rendering/scene_render_items.hpp>
#include <gobot/scene/camera_3d.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/environment_3d.hpp>
#include <gobot/scene/light_3d.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/terrain_3d.hpp>
#include <gobot/scene/window.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/resources/box_shape_3d.hpp>
#include <gobot/scene/resources/array_mesh.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>

TEST(TestMeshInstance3D, render_server_forwards_backend_neutral_renderer_settings) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    gobot::SceneRendererSettings settings;
    settings.mode = gobot::SceneRendererMode::ProgressivePathTracing;
    settings.target_fps = 45;
    settings.samples_per_frame = 8;
    settings.max_bounces = 6;
    settings.denoise = false;
    render_server->SetSceneRendererSettings(settings);

    const gobot::SceneRendererSettings stored = render_server->GetSceneRendererSettings();
    EXPECT_EQ(stored.mode, gobot::SceneRendererMode::ProgressivePathTracing);
    EXPECT_EQ(stored.target_fps, 45);
    EXPECT_EQ(stored.samples_per_frame, 8);
    EXPECT_EQ(stored.max_bounces, 6);
    EXPECT_FALSE(stored.denoise);

    const gobot::SceneRendererCapabilities capabilities =
            render_server->GetSceneRendererCapabilities();
    EXPECT_FALSE(capabilities.ray_tracing_available);
}

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

TEST(TestMeshInstance3D, render_snapshot_captures_visible_meshes_and_camera) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    auto* mesh_instance = gobot::Object::New<gobot::MeshInstance3D>();
    mesh_instance->SetMesh(gobot::MakeRef<gobot::BoxMesh>());
    tree->GetRoot()->AddChild(mesh_instance, true);

    gobot::Camera3D camera;
    camera.SetAspect(16.0 / 9.0);
    camera.SetPerspective(60.0, 0.1, 500.0);
    camera.SetViewMatrix({3.0, -4.0, 2.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0});

    const gobot::SceneRenderSnapshot snapshot =
            gobot::CaptureSceneRenderSnapshot(tree->GetRoot(), camera);
    ASSERT_EQ(snapshot.visual_meshes.size(), 1);
    ASSERT_NE(snapshot.visual_meshes[0].GetSurface(), nullptr);
    EXPECT_TRUE(snapshot.visual_meshes[0].mesh_id.IsValid());
    EXPECT_GT(snapshot.visual_meshes[0].mesh_revision, 0);
    EXPECT_TRUE(snapshot.camera.view.isApprox(camera.GetViewMatrix(), CMP_EPSILON));
    EXPECT_TRUE(snapshot.camera.projection.isApprox(camera.GetProjectionMatrix(), CMP_EPSILON));
    EXPECT_TRUE(snapshot.camera.view_projection.isApprox(
            camera.GetProjectionMatrix() * camera.GetViewMatrix(), CMP_EPSILON));
    EXPECT_TRUE(snapshot.camera.world_position.isApprox(
            gobot::Vector3(3.0, -4.0, 2.0), CMP_EPSILON));
    EXPECT_GT(snapshot.environment.directional_light_intensity, 0.0);
    EXPECT_GT(snapshot.environment.ambient_intensity, 0.0);
    ASSERT_EQ(snapshot.lights.size(), 1);
    EXPECT_NE(snapshot.fingerprints.geometry, 0);
    EXPECT_NE(snapshot.fingerprints.camera, 0);
    EXPECT_NE(snapshot.fingerprints.combined, 0);

    gobot::Object::Delete(tree);
}

TEST(TestMeshInstance3D, render_snapshot_captures_authored_environment_and_lights) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    auto* environment = gobot::Object::New<gobot::Environment3D>();
    environment->SetSkyColor({0.1f, 0.2f, 0.4f, 1.0f});
    environment->SetAmbientIntensity(0.7);
    environment->SetExposure(1.25);
    tree->GetRoot()->AddChild(environment, true);

    auto* directional = gobot::Object::New<gobot::DirectionalLight3D>();
    directional->SetIntensity(2.5);
    directional->SetColor({1.0f, 0.8f, 0.6f, 1.0f});
    tree->GetRoot()->AddChild(directional, true);

    auto* point = gobot::Object::New<gobot::PointLight3D>();
    point->SetPosition({1.0, 2.0, 3.0});
    point->SetRange(12.0);
    tree->GetRoot()->AddChild(point, true);

    auto* spot = gobot::Object::New<gobot::SpotLight3D>();
    spot->SetInnerAngle(15.0);
    spot->SetOuterAngle(35.0);
    tree->GetRoot()->AddChild(spot, true);

    gobot::Camera3D camera;
    camera.SetPerspective(60.0, 0.1, 100.0);
    camera.SetViewMatrix({3.0, -3.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
    const gobot::SceneRenderSnapshot snapshot =
            gobot::CaptureSceneRenderSnapshot(tree->GetRoot(), camera);

    EXPECT_FLOAT_EQ(snapshot.environment.sky_color.blue(), 0.4f);
    EXPECT_FLOAT_EQ(snapshot.environment.ambient_intensity, 0.7f);
    EXPECT_FLOAT_EQ(snapshot.environment.exposure, 1.25f);
    ASSERT_EQ(snapshot.lights.size(), 3);
    EXPECT_EQ(snapshot.lights[0].type, gobot::RenderLightType::Directional);
    EXPECT_FLOAT_EQ(snapshot.lights[0].intensity, 2.5f);
    EXPECT_EQ(snapshot.lights[1].type, gobot::RenderLightType::Point);
    EXPECT_FLOAT_EQ(snapshot.lights[1].range, 12.0f);
    EXPECT_EQ(snapshot.lights[2].type, gobot::RenderLightType::Spot);
    EXPECT_FLOAT_EQ(snapshot.lights[2].outer_angle, 35.0f);

    gobot::Object::Delete(tree);
}

TEST(TestMeshInstance3D, render_snapshot_fingerprints_isolate_scene_changes) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    auto* tree = gobot::Object::New<gobot::SceneTree>(false);
    tree->Initialize();

    auto material = gobot::MakeRef<gobot::PBRMaterial3D>();
    auto* mesh_instance = gobot::Object::New<gobot::MeshInstance3D>();
    mesh_instance->SetMesh(gobot::MakeRef<gobot::BoxMesh>());
    mesh_instance->SetMaterial(material);
    tree->GetRoot()->AddChild(mesh_instance, true);

    gobot::Camera3D camera;
    camera.SetPerspective(60.0, 0.1, 100.0);
    camera.SetViewMatrix({3.0, -3.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
    const auto baseline = gobot::CaptureSceneRenderSnapshot(tree->GetRoot(), camera);

    material->SetRoughness(0.2);
    const auto material_changed = gobot::CaptureSceneRenderSnapshot(tree->GetRoot(), camera);
    EXPECT_EQ(material_changed.fingerprints.geometry, baseline.fingerprints.geometry);
    EXPECT_EQ(material_changed.fingerprints.transforms, baseline.fingerprints.transforms);
    EXPECT_NE(material_changed.fingerprints.materials, baseline.fingerprints.materials);

    mesh_instance->SetPosition({1.0, 0.0, 0.0});
    const auto transform_changed = gobot::CaptureSceneRenderSnapshot(tree->GetRoot(), camera);
    EXPECT_NE(transform_changed.fingerprints.transforms, material_changed.fingerprints.transforms);
    EXPECT_EQ(transform_changed.fingerprints.materials, material_changed.fingerprints.materials);

    camera.SetViewMatrix({4.0, -3.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0});
    const auto camera_changed = gobot::CaptureSceneRenderSnapshot(tree->GetRoot(), camera);
    EXPECT_NE(camera_changed.fingerprints.camera, transform_changed.fingerprints.camera);
    EXPECT_EQ(camera_changed.fingerprints.geometry, transform_changed.fingerprints.geometry);

    gobot::Object::Delete(tree);
}

TEST(TestMeshInstance3D, render_scene_snapshot_assigns_stable_instance_and_inherited_semantic_ids) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("World");
    root->SetSemanticLabel("robot");
    auto* link = gobot::Object::New<gobot::Node3D>();
    link->SetName("link");
    auto* mesh_instance = gobot::Object::New<gobot::MeshInstance3D>();
    mesh_instance->SetName("visual");

    gobot::MeshSurfaceData first;
    first.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    first.indices = {0, 1, 2};
    gobot::MeshSurfaceData second = first;
    second.vertices = {{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}};
    auto mesh = gobot::MakeRef<gobot::ArrayMesh>();
    mesh->SetSurfaces({first, second});
    mesh_instance->SetMesh(mesh);
    link->AddChild(mesh_instance);
    root->AddChild(link);

    const gobot::RenderSceneSnapshot first_snapshot = gobot::CaptureRenderSceneSnapshot(root);
    ASSERT_EQ(first_snapshot.visual_meshes.size(), 2);
    EXPECT_EQ(first_snapshot.visual_meshes[0].instance_id,
              first_snapshot.visual_meshes[1].instance_id);
    EXPECT_EQ(first_snapshot.visual_meshes[0].semantic_id,
              first_snapshot.visual_meshes[1].semantic_id);
    EXPECT_EQ(first_snapshot.visual_meshes[0].instance_path, "link/visual");
    EXPECT_EQ(first_snapshot.visual_meshes[0].semantic_label, "robot");
    EXPECT_EQ(first_snapshot.instance_paths.at(first_snapshot.visual_meshes[0].instance_id),
              "link/visual");
    EXPECT_EQ(first_snapshot.semantic_labels.at(first_snapshot.visual_meshes[0].semantic_id),
              "robot");

    gobot::Ref<gobot::PackedScene> packed = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed->Pack(root));
    gobot::Node* restored = packed->Instantiate();
    ASSERT_NE(restored, nullptr);
    const gobot::RenderSceneSnapshot restored_snapshot = gobot::CaptureRenderSceneSnapshot(restored);
    ASSERT_EQ(restored_snapshot.visual_meshes.size(), 2);
    EXPECT_EQ(restored_snapshot.visual_meshes[0].instance_id,
              first_snapshot.visual_meshes[0].instance_id);
    EXPECT_EQ(restored_snapshot.visual_meshes[0].semantic_id,
              first_snapshot.visual_meshes[0].semantic_id);
    EXPECT_EQ(restored_snapshot.instance_paths, first_snapshot.instance_paths);
    EXPECT_EQ(restored_snapshot.semantic_labels, first_snapshot.semantic_labels);

    gobot::Object::Delete(root);
    gobot::Object::Delete(restored);
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
    EXPECT_FLOAT_EQ(items.visual_meshes[0].material.albedo.red(), 1.0f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].material.albedo.green(), 1.0f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].material.albedo.blue(), 1.0f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].material.albedo.alpha(), 0.75f);

    terrain->SetColorMode(gobot::TerrainColorMode::SurfaceColor);
    items = gobot::CollectSceneRenderItems(tree->GetRoot());
    ASSERT_EQ(items.visual_meshes.size(), 1);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].material.albedo.red(), 0.2f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].material.albedo.green(), 0.3f);
    EXPECT_FLOAT_EQ(items.visual_meshes[0].material.albedo.blue(), 0.4f);

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
