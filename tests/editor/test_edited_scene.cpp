/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
*/

#include <gtest/gtest.h>

#include <filesystem>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/resource_format_scene.hpp>
#include <gobot/core/io/resource_format_urdf.hpp>
#include <gobot/editor/edited_scene.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/robot_3d.hpp>

class TestEditedScene : public testing::Test {
protected:
    static void SetUpTestSuite() {
        static gobot::Ref<gobot::ResourceFormatLoaderScene> resource_loader_scene;
        static gobot::Ref<gobot::ResourceFormatLoaderURDF> resource_loader_urdf;
        static gobot::Ref<gobot::ResourceFormatSaverScene> resource_saver_scene;

        resource_saver_scene = gobot::MakeRef<gobot::ResourceFormatSaverScene>();
        gobot::ResourceSaver::AddResourceFormatSaver(resource_saver_scene, true);

        resource_loader_scene = gobot::MakeRef<gobot::ResourceFormatLoaderScene>();
        gobot::ResourceLoader::AddResourceFormatLoader(resource_loader_scene, true);

        resource_loader_urdf = gobot::MakeRef<gobot::ResourceFormatLoaderURDF>();
        gobot::ResourceLoader::AddResourceFormatLoader(resource_loader_urdf, true);
    }

    void SetUp() override {
        std::filesystem::create_directories("/tmp/test_project");
        gobot::ProjectSettings::GetInstance()->SetProjectPath("/tmp/test_project");
    }

    gobot::ProjectSettings project_settings;
};

TEST_F(TestEditedScene, saves_and_loads_user_scene_root) {
    auto* edited_scene = gobot::Object::New<gobot::EditedScene>();
    edited_scene->GetRoot()->SetName("RobotRoot");
    edited_scene->GetRoot()->SetPosition({1.0f, 2.0f, 3.0f});

    auto* arm = gobot::Object::New<gobot::Node3D>();
    arm->SetName("Arm");
    arm->SetPosition({4.0f, 5.0f, 6.0f});
    edited_scene->GetRoot()->AddChild(arm);

    ASSERT_TRUE(edited_scene->SaveToPath("res://edited_scene.jscn"));

    auto* loaded_scene = gobot::Object::New<gobot::EditedScene>();
    ASSERT_TRUE(loaded_scene->LoadFromPath("res://edited_scene.jscn"));

    gobot::Node3D* loaded_root = loaded_scene->GetRoot();
    ASSERT_NE(loaded_root, nullptr);
    EXPECT_EQ(loaded_root->GetName(), "RobotRoot");
    EXPECT_TRUE(loaded_root->GetPosition().isApprox(gobot::Vector3(1.0f, 2.0f, 3.0f), CMP_EPSILON));
    ASSERT_EQ(loaded_root->GetChildCount(), 1);

    auto* loaded_arm = gobot::Object::PointerCastTo<gobot::Node3D>(loaded_root->GetChild(0));
    ASSERT_NE(loaded_arm, nullptr);
    EXPECT_EQ(loaded_arm->GetName(), "Arm");
    EXPECT_TRUE(loaded_arm->GetPosition().isApprox(gobot::Vector3(4.0f, 5.0f, 6.0f), CMP_EPSILON));

    gobot::Object::Delete(edited_scene);
    gobot::Object::Delete(loaded_scene);
}

TEST_F(TestEditedScene, loads_urdf_as_editable_robot_scene_tree) {
    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/urdf/simple_robot.urdf";

    auto* edited_scene = gobot::Object::New<gobot::EditedScene>();
    ASSERT_TRUE(edited_scene->LoadFromPath(fixture_path.string()));

    auto* robot = gobot::Object::PointerCastTo<gobot::Robot3D>(edited_scene->GetRoot());
    ASSERT_NE(robot, nullptr);
    EXPECT_EQ(robot->GetName(), "test_bot");
    ASSERT_EQ(robot->GetChildCount(), 1);

    auto* base = gobot::Object::PointerCastTo<gobot::Link3D>(robot->GetChild(0));
    ASSERT_NE(base, nullptr);
    EXPECT_EQ(base->GetName(), "base_link");
    ASSERT_EQ(base->GetChildCount(), 3);

    EXPECT_NE(gobot::Object::PointerCastTo<gobot::MeshInstance3D>(base->GetChild(0)), nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::CollisionShape3D>(base->GetChild(1)), nullptr);

    gobot::Object::Delete(edited_scene);
}
