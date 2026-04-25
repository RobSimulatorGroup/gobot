/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include <gtest/gtest.h>

#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/scene_tree.hpp>
#include <gobot/scene/window.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
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

TEST(TestMeshInstance3D, node3d_transform_properties_are_reflected) {
    auto position = gobot::Type::get<gobot::Node3D>().get_property("position");
    auto rotation = gobot::Type::get<gobot::Node3D>().get_property("rotation_degrees");
    auto scale = gobot::Type::get<gobot::Node3D>().get_property("scale");
    auto mesh = gobot::Type::get<gobot::MeshInstance3D>().get_property("mesh");

    EXPECT_TRUE(position.is_valid());
    EXPECT_TRUE(rotation.is_valid());
    EXPECT_TRUE(scale.is_valid());
    EXPECT_TRUE(mesh.is_valid());
}
