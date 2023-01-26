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
#include <gobot/scene/resources/primitive_mesh.hpp>
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
        auto* project_setting = gobot::ProjectSettings::GetSingleton();
        project_setting->SetProjectPath("/tmp/test_project");
    }

    void TearDown() override {
    }

    gobot::ProjectSettings project_settings;
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
    auto material_3d = gobot::MakeRef<gobot::Material3D>();
    box_mesh->SetMaterial(material_3d);


    LOG_ERROR("{}", material_3d.use_count());

    USING_ENUM_BITWISE_OPERATORS;
    gobot::ResourceSaver::Save(box_mesh, "res://box_mesh.jres",
                               gobot::ResourceSaverFlags::ReplaceSubResourcePaths |
                               gobot::ResourceSaverFlags::ChangePath);
    LOG_ERROR("{}", box_mesh->GetMaterial() == material_3d);
    LOG_ERROR("{}", material_3d.use_count());

//    gobot::Ref<gobot::Resource> cylinder = gobot::ResourceLoader::Load("res://cyl.jres");
//    ASSERT_TRUE(cylinder->get_type().get_name() == "CylinderShape3D");
//    cy = gobot::dynamic_pointer_cast<gobot::CylinderShape3D>(cylinder);
//    ASSERT_TRUE(cy->GetRadius() ==  1.1f);
}

//TEST(TestResourceFormatScene, ext_subresource) {
//
//}