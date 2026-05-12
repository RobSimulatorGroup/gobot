/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/resource_format_scene.hpp>
#include <gobot/core/io/resource_format_urdf.hpp>
#include <gobot/editor/edited_scene.hpp>
#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/collision_shape_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>
#include <gobot/scene/resources/material.hpp>
#include <gobot/scene/robot_3d.hpp>

namespace {

gobot::Node* FindNodeByName(gobot::Node* node, const std::string& name) {
    if (node == nullptr) {
        return nullptr;
    }
    if (node->GetName() == name) {
        return node;
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        if (auto* found = FindNodeByName(node->GetChild(static_cast<int>(i)), name)) {
            return found;
        }
    }
    return nullptr;
}

} // namespace

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
        render_server = std::make_unique<gobot::RenderServer>();
    }

    void TearDown() override {
        render_server.reset();
    }

    gobot::ProjectSettings project_settings;
    std::unique_ptr<gobot::RenderServer> render_server;
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

TEST_F(TestEditedScene, loads_urdf_with_default_motion_mode_when_requested) {
    const std::filesystem::path fixture_path =
            std::filesystem::current_path() / "tests/fixtures/urdf/simple_robot.urdf";

    auto* edited_scene = gobot::Object::New<gobot::EditedScene>();
    ASSERT_TRUE(edited_scene->LoadFromPath(fixture_path.string(), true));

    auto* robot = gobot::Object::PointerCastTo<gobot::Robot3D>(edited_scene->GetRoot());
    ASSERT_NE(robot, nullptr);
    EXPECT_EQ(robot->GetMode(), gobot::RobotMode::Motion);

    auto* joint = gobot::Object::PointerCastTo<gobot::Joint3D>(
            FindNodeByName(robot, "shoulder_pan_joint"));
    ASSERT_NE(joint, nullptr);
    EXPECT_TRUE(joint->IsMotionModeEnabled());

    gobot::Object::Delete(edited_scene);
}

TEST_F(TestEditedScene, robot_scene_round_trips_motion_state_and_materials) {
    auto* edited_scene = gobot::Object::New<gobot::EditedScene>();
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("Robot");
    robot->SetSourcePath("res://robot.urdf");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    robot->AddChild(base);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetParentLink("base");
    joint->SetChildLink("body");
    joint->SetAxis(gobot::Vector3::UnitZ());
    joint->SetLowerLimit(1.0f);
    joint->SetUpperLimit(3.0f);
    joint->SetJointPosition(2.25f);
    base->AddChild(joint);

    auto* body = gobot::Object::New<gobot::Link3D>();
    body->SetName("body");
    body->SetPosition({1.0f, 0.0f, 0.0f});
    joint->AddChild(body);

    auto* visual = gobot::Object::New<gobot::MeshInstance3D>();
    visual->SetName("visual");
    auto box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetSize({0.5f, 0.6f, 0.7f});
    auto material = gobot::MakeRef<gobot::PBRMaterial3D>();
    material->SetAlbedo({0.25f, 0.5f, 0.75f, 1.0f});
    material->SetMetallic(0.4f);
    material->SetRoughness(0.65f);
    box_mesh->SetMaterial(material);
    visual->SetMesh(box_mesh);
    body->AddChild(visual);

    robot->SetMode(gobot::RobotMode::Motion);
    const gobot::Vector3 expected_motion_body_position = body->GetPosition();
    edited_scene->GetRoot()->AddChild(robot);
    ASSERT_TRUE(edited_scene->SaveToPath("res://robot_motion_roundtrip.jscn"));

    auto* loaded_scene = gobot::Object::New<gobot::EditedScene>();
    ASSERT_TRUE(loaded_scene->LoadFromPath("res://robot_motion_roundtrip.jscn"));

    auto* loaded_robot = gobot::Object::PointerCastTo<gobot::Robot3D>(
            FindNodeByName(loaded_scene->GetRoot(), "Robot"));
    ASSERT_NE(loaded_robot, nullptr);
    EXPECT_EQ(loaded_robot->GetMode(), gobot::RobotMode::Motion);
    EXPECT_EQ(loaded_robot->GetSourcePath(), "res://robot.urdf");

    auto* loaded_joint = gobot::Object::PointerCastTo<gobot::Joint3D>(
            FindNodeByName(loaded_robot, "joint"));
    ASSERT_NE(loaded_joint, nullptr);
    EXPECT_EQ(loaded_joint->GetJointType(), gobot::JointType::Revolute);
    EXPECT_FLOAT_EQ(loaded_joint->GetLowerLimit(), 1.0f);
    EXPECT_FLOAT_EQ(loaded_joint->GetUpperLimit(), 3.0f);
    EXPECT_FLOAT_EQ(loaded_joint->GetJointPosition(), 2.25f);
    EXPECT_TRUE(loaded_joint->IsMotionModeEnabled());

    auto* loaded_body = gobot::Object::PointerCastTo<gobot::Link3D>(
            FindNodeByName(loaded_robot, "body"));
    ASSERT_NE(loaded_body, nullptr);
    EXPECT_TRUE(loaded_body->GetPosition().isApprox(expected_motion_body_position, CMP_EPSILON));

    auto* loaded_visual = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(
            FindNodeByName(loaded_robot, "visual"));
    ASSERT_NE(loaded_visual, nullptr);
    gobot::Ref<gobot::BoxMesh> loaded_mesh =
            gobot::dynamic_pointer_cast<gobot::BoxMesh>(loaded_visual->GetMesh());
    ASSERT_TRUE(loaded_mesh.IsValid());
    EXPECT_TRUE(loaded_mesh->GetSize().isApprox(gobot::Vector3(0.5f, 0.6f, 0.7f), CMP_EPSILON));

    gobot::Ref<gobot::PBRMaterial3D> loaded_material =
            gobot::dynamic_pointer_cast<gobot::PBRMaterial3D>(loaded_mesh->GetMaterial());
    ASSERT_TRUE(loaded_material.IsValid());
    const gobot::Color loaded_albedo = loaded_material->GetAlbedo();
    EXPECT_FLOAT_EQ(loaded_albedo.red(), 0.25f);
    EXPECT_FLOAT_EQ(loaded_albedo.green(), 0.5f);
    EXPECT_FLOAT_EQ(loaded_albedo.blue(), 0.75f);
    EXPECT_FLOAT_EQ(loaded_albedo.alpha(), 1.0f);
    EXPECT_FLOAT_EQ(loaded_material->GetMetallic(), 0.4f);
    EXPECT_FLOAT_EQ(loaded_material->GetRoughness(), 0.65f);

    gobot::Object::Delete(edited_scene);
    gobot::Object::Delete(loaded_scene);
}

TEST_F(TestEditedScene, robot_scene_restores_joint_position_after_joint_limits_from_json_order) {
    auto* edited_scene = gobot::Object::New<gobot::EditedScene>();
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("Robot");

    auto* base = gobot::Object::New<gobot::Link3D>();
    base->SetName("base");
    robot->AddChild(base);

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetAxis(gobot::Vector3::UnitZ());
    joint->SetLowerLimit(1.0f);
    joint->SetUpperLimit(3.0f);
    joint->SetJointPosition(2.0f);
    base->AddChild(joint);

    auto* body = gobot::Object::New<gobot::Link3D>();
    body->SetName("body");
    body->SetPosition({1.0f, 0.0f, 0.0f});
    joint->AddChild(body);

    robot->SetMode(gobot::RobotMode::Motion);
    const gobot::Vector3 expected_motion_body_position = body->GetPosition();
    edited_scene->GetRoot()->AddChild(robot);
    ASSERT_TRUE(edited_scene->SaveToPath("res://robot_joint_order_roundtrip.jscn"));

    auto* loaded_scene = gobot::Object::New<gobot::EditedScene>();
    ASSERT_TRUE(loaded_scene->LoadFromPath("res://robot_joint_order_roundtrip.jscn"));

    auto* loaded_joint = gobot::Object::PointerCastTo<gobot::Joint3D>(
            FindNodeByName(loaded_scene->GetRoot(), "joint"));
    ASSERT_NE(loaded_joint, nullptr);
    EXPECT_EQ(loaded_joint->GetJointType(), gobot::JointType::Revolute);
    EXPECT_FLOAT_EQ(loaded_joint->GetLowerLimit(), 1.0f);
    EXPECT_FLOAT_EQ(loaded_joint->GetUpperLimit(), 3.0f);
    EXPECT_FLOAT_EQ(loaded_joint->GetJointPosition(), 2.0f);
    EXPECT_TRUE(loaded_joint->IsMotionModeEnabled());

    auto* loaded_body = gobot::Object::PointerCastTo<gobot::Link3D>(
            FindNodeByName(loaded_scene->GetRoot(), "body"));
    ASSERT_NE(loaded_body, nullptr);
    EXPECT_TRUE(loaded_body->GetPosition().isApprox(expected_motion_body_position, CMP_EPSILON));

    gobot::Object::Delete(edited_scene);
    gobot::Object::Delete(loaded_scene);
}

TEST_F(TestEditedScene, creates_new_scene_and_adds_saved_robot_scene_as_child) {
    auto* robot_scene = gobot::Object::New<gobot::EditedScene>();
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("RobotAsset");
    robot_scene->GetRoot()->AddChild(robot);
    ASSERT_TRUE(robot_scene->SaveToPath("res://robot_asset.jscn"));

    auto* world_scene = gobot::Object::New<gobot::EditedScene>();
    ASSERT_TRUE(world_scene->NewScene());
    ASSERT_NE(world_scene->GetRoot(), nullptr);
    EXPECT_EQ(world_scene->GetRoot()->GetName(), "Scene");
    ASSERT_EQ(world_scene->GetRoot()->GetChildCount(), 0);

    gobot::Node3D* added_scene = world_scene->AddSceneFromPath("res://robot_asset.jscn");
    ASSERT_NE(added_scene, nullptr);
    EXPECT_EQ(added_scene->GetParent(), world_scene->GetRoot());
    ASSERT_EQ(world_scene->GetRoot()->GetChildCount(), 1);
    EXPECT_NE(FindNodeByName(world_scene->GetRoot(), "RobotAsset"), nullptr);

    gobot::Object::Delete(robot_scene);
    gobot::Object::Delete(world_scene);
}
