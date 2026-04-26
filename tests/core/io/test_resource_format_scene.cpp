/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
*/

#include <gtest/gtest.h>

#include <gobot/core/ref_counted.hpp>
#include <gobot/core/io/resource_format_scene.hpp>
#include <gobot/core/config/project_setting.hpp>
#include <gobot/scene/resources/cylinder_shape_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>


class TestResourceFormatScene : public testing::Test {
protected:
    static void SetUpTestSuite() {
        static gobot::Ref<gobot::ResourceFormatLoaderScene> resource_loader_scene;
        static gobot::Ref<gobot::ResourceFormatSaverScene> resource_saver_scene;

        resource_saver_scene = gobot::MakeRef<gobot::ResourceFormatSaverScene>();
        gobot::ResourceSaver::AddResourceFormatSaver(resource_saver_scene, true);

        resource_loader_scene = gobot::MakeRef<gobot::ResourceFormatLoaderScene>();
        gobot::ResourceLoader::AddResourceFormatLoader(resource_loader_scene, true);
    }

    static void TearDownTestSuite() {
    }

    void SetUp() override {
        auto* project_setting = gobot::ProjectSettings::GetInstance();
        project_setting->SetProjectPath("/tmp/test_project");

        render_server = std::make_unique<gobot::RenderServer>();
    }

    void TearDown() override {
    }

    gobot::ProjectSettings project_settings;
    std::unique_ptr<gobot::RenderServer> render_server;
};

TEST_F(TestResourceFormatScene, test_save_load) {
    gobot::Ref<gobot::CylinderShape3D> cy = gobot::MakeRef<gobot::CylinderShape3D>();
    cy->SetRadius(1.1);

    USING_ENUM_BITWISE_OPERATORS;
    gobot::ResourceSaver::Save(cy, "res://cyl.jres",
                               gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                               gobot::ResourceSaverFlags::ChangePath);

    gobot::Ref<gobot::Resource> cylinder = gobot::ResourceLoader::Load("res://cyl.jres");
    ASSERT_TRUE(cylinder->get_type().get_name() == "CylinderShape3D");
    cy = gobot::dynamic_pointer_cast<gobot::CylinderShape3D>(cylinder);
    ASSERT_TRUE(cy->GetRadius() ==  1.1f);
}

TEST_F(TestResourceFormatScene, test_subresource) {
    gobot::Ref<gobot::BoxMesh> box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetWidth(1.1);
    auto material_3d = gobot::MakeRef<gobot::PBRMaterial3D>();
    material_3d->SetAlbedo(gobot::Color(0.5f, 0.5f, 0.1f));
    box_mesh->SetMaterial(material_3d);


    USING_ENUM_BITWISE_OPERATORS;
    gobot::ResourceSaver::Save(box_mesh, "res://box_mesh.jres",
                               gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                               gobot::ResourceSaverFlags::ChangePath);

    gobot::Ref<gobot::Resource> box = gobot::ResourceLoader::Load("res://box_mesh.jres");
    ASSERT_TRUE(box->get_type().get_name() == "BoxMesh");
    auto box_load = gobot::dynamic_pointer_cast<gobot::BoxMesh>(box);
    ASSERT_TRUE(box_load->GetWidth() ==  1.1f);
    auto material3d = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(box_load->GetMaterial());
    ASSERT_TRUE(material3d.IsValid());
    ASSERT_TRUE(material3d->GetAlbedo().blue() == 0.1f);
}

TEST_F(TestResourceFormatScene, test_extresource) {
    gobot::Ref<gobot::BoxMesh> box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetWidth(1.1);
    auto material_3d = gobot::MakeRef<gobot::PBRMaterial3D>();
    material_3d->SetAlbedo(gobot::Color(0.5f, 0.5f, 0.1f));
    material_3d->SetPath("res://material_ext.jres", true);
    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(material_3d, "res://material_ext.jres",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths));


    box_mesh->SetMaterial(material_3d);

    ASSERT_TRUE(gobot::ResourceSaver::Save(box_mesh, "res://box_mesh_ext.jres",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths));
    material_3d->SetPath("", true);

    gobot::Ref<gobot::Resource> box = gobot::ResourceLoader::Load("res://box_mesh_ext.jres");
    ASSERT_TRUE(box->get_type().get_name() == "BoxMesh");
    auto box_load = gobot::dynamic_pointer_cast<gobot::BoxMesh>(box);
    ASSERT_TRUE(box_load->GetWidth() ==  1.1f);
    auto material3d = gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(box_load->GetMaterial());
    ASSERT_TRUE(material3d.IsValid());
    ASSERT_TRUE(material3d->GetAlbedo().blue() == 0.1f);
}

TEST_F(TestResourceFormatScene, packed_scene_round_trips_nodes_transforms_and_mesh_resource) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("RobotRoot");
    root->SetPosition({1.0f, 2.0f, 3.0f});

    auto* box = gobot::Object::New<gobot::MeshInstance3D>();
    box->SetName("Box");
    box->SetPosition({4.0f, 5.0f, 6.0f});

    gobot::Ref<gobot::BoxMesh> box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetSize({2.0f, 3.0f, 4.0f});
    box->SetMesh(box_mesh);
    root->AddChild(box);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    USING_ENUM_BITWISE_OPERATORS;
    ASSERT_TRUE(gobot::ResourceSaver::Save(packed_scene, "res://robot_scene.jscn",
                                           gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                                           gobot::ResourceSaverFlags::ChangePath));

    gobot::Ref<gobot::Resource> loaded_resource = gobot::ResourceLoader::Load(
            "res://robot_scene.jscn", "PackedScene", gobot::ResourceFormatLoader::CacheMode::Ignore);
    ASSERT_TRUE(loaded_resource.IsValid());

    gobot::Ref<gobot::PackedScene> loaded_scene = gobot::dynamic_pointer_cast<gobot::PackedScene>(loaded_resource);
    ASSERT_TRUE(loaded_scene.IsValid());
    ASSERT_EQ(loaded_scene->GetState()->GetNodeCount(), 2);

    gobot::Node* instance = loaded_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->GetName(), "RobotRoot");

    auto* instanced_root = gobot::Object::PointerCastTo<gobot::Node3D>(instance);
    ASSERT_NE(instanced_root, nullptr);
    EXPECT_TRUE(instanced_root->GetPosition().isApprox(gobot::Vector3(1.0f, 2.0f, 3.0f), CMP_EPSILON));
    ASSERT_EQ(instanced_root->GetChildCount(), 1);

    auto* instanced_box = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(instanced_root->GetChild(0));
    ASSERT_NE(instanced_box, nullptr);
    EXPECT_EQ(instanced_box->GetName(), "Box");
    EXPECT_TRUE(instanced_box->GetPosition().isApprox(gobot::Vector3(4.0f, 5.0f, 6.0f), CMP_EPSILON));

    gobot::Ref<gobot::BoxMesh> instanced_mesh = gobot::dynamic_pointer_cast<gobot::BoxMesh>(instanced_box->GetMesh());
    ASSERT_TRUE(instanced_mesh.IsValid());
    EXPECT_TRUE(instanced_mesh->GetSize().isApprox(gobot::Vector3(2.0f, 3.0f, 4.0f), CMP_EPSILON));

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}
