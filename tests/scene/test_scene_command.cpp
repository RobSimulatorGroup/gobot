#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/python_script.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/object.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/scene_command.hpp"

namespace {

class TestSceneCommand : public ::testing::Test {
protected:
    void SetUp() override {
        setenv("HOME", "/tmp/gobot-test-home", 1);
    }

    void TearDown() override {
        if (root != nullptr) {
            gobot::Object::Delete(root);
            root = nullptr;
        }
        if (detached != nullptr && detached->GetParent() == nullptr) {
            gobot::Object::Delete(detached);
            detached = nullptr;
        }
    }

    gobot::Node3D* root{nullptr};
    gobot::Node3D* detached{nullptr};
};

} // namespace

TEST_F(TestSceneCommand, tracks_dirty_clean_undo_and_redo) {
    root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    gobot::SceneCommandStack stack;
    stack.MarkClean();
    ASSERT_FALSE(stack.IsDirty());

    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::SetNodePropertyCommand>(
            root->GetInstanceId(),
            "position",
            gobot::Variant(gobot::Vector3(1.0, 2.0, 3.0)))));

    EXPECT_TRUE(stack.IsDirty());
    EXPECT_TRUE(root->GetPosition().isApprox(gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));

    ASSERT_TRUE(stack.Undo());
    EXPECT_FALSE(stack.IsDirty());
    EXPECT_TRUE(root->GetPosition().isApprox(gobot::Vector3::Zero(), CMP_EPSILON));

    ASSERT_TRUE(stack.Redo());
    EXPECT_TRUE(stack.IsDirty());
    EXPECT_TRUE(root->GetPosition().isApprox(gobot::Vector3(1.0, 2.0, 3.0), CMP_EPSILON));
}

TEST_F(TestSceneCommand, transaction_undoes_as_one_command) {
    root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    gobot::SceneCommandStack stack;
    stack.MarkClean();
    ASSERT_TRUE(stack.BeginTransaction("Move and rename"));
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::RenameNodeCommand>(root->GetInstanceId(), "renamed")));
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::SetNodePropertyCommand>(
            root->GetInstanceId(),
            "position",
            gobot::Variant(gobot::Vector3(4.0, 5.0, 6.0)))));
    ASSERT_TRUE(stack.CommitTransaction());

    EXPECT_EQ(root->GetName(), "renamed");
    EXPECT_TRUE(root->GetPosition().isApprox(gobot::Vector3(4.0, 5.0, 6.0), CMP_EPSILON));

    ASSERT_TRUE(stack.Undo());
    EXPECT_EQ(root->GetName(), "root");
    EXPECT_TRUE(root->GetPosition().isApprox(gobot::Vector3::Zero(), CMP_EPSILON));

    ASSERT_TRUE(stack.Redo());
    EXPECT_EQ(root->GetName(), "renamed");
    EXPECT_TRUE(root->GetPosition().isApprox(gobot::Vector3(4.0, 5.0, 6.0), CMP_EPSILON));
}

TEST_F(TestSceneCommand, cancelled_transaction_reverts_executed_commands) {
    root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");

    gobot::SceneCommandStack stack;
    ASSERT_TRUE(stack.BeginTransaction("Cancel me"));
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::RenameNodeCommand>(root->GetInstanceId(), "cancelled")));
    ASSERT_TRUE(stack.CancelTransaction());

    EXPECT_EQ(root->GetName(), "root");
    EXPECT_FALSE(stack.CanUndo());
}

TEST_F(TestSceneCommand, delete_undo_restores_new_node_without_reviving_old_id) {
    root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");
    detached = gobot::Object::New<gobot::Node3D>();
    detached->SetName("child");
    root->AddChild(detached);

    const gobot::ObjectID old_child_id = detached->GetInstanceId();
    gobot::SceneCommandStack stack;
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::RemoveChildNodeCommand>(
            root->GetInstanceId(),
            detached->GetInstanceId(),
            true)));
    detached = nullptr;

    EXPECT_EQ(root->GetChildCount(), 0);
    EXPECT_EQ(gobot::ObjectDB::GetInstance(old_child_id), nullptr);

    ASSERT_TRUE(stack.Undo());
    ASSERT_EQ(root->GetChildCount(), 1);
    gobot::Node* restored = root->GetChild(0);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->GetName(), "child");
    EXPECT_NE(restored->GetInstanceId(), old_child_id);
    EXPECT_EQ(gobot::ObjectDB::GetInstance(old_child_id), nullptr);
}

TEST_F(TestSceneCommand, numeric_property_commands_convert_to_reflected_property_type) {
    root = gobot::Object::New<gobot::Node3D>();
    root->SetName("root");
    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("slider");
    joint->SetJointType(gobot::JointType::Prismatic);
    root->AddChild(joint);

    gobot::SceneCommandStack stack;
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::SetNodePropertyCommand>(
            joint->GetInstanceId(),
            "lower_limit",
            gobot::Variant(-2.0f))));
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::SetNodePropertyCommand>(
            joint->GetInstanceId(),
            "upper_limit",
            gobot::Variant(2.0f))));
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::SetNodePropertyCommand>(
            joint->GetInstanceId(),
            "joint_position",
            gobot::Variant(0.5f))));

    EXPECT_NEAR(joint->GetLowerLimit(), -2.0, CMP_EPSILON);
    EXPECT_NEAR(joint->GetUpperLimit(), 2.0, CMP_EPSILON);
    EXPECT_NEAR(joint->GetJointPosition(), 0.5, CMP_EPSILON);

    ASSERT_TRUE(stack.Undo());
    EXPECT_NEAR(joint->GetJointPosition(), 0.0, CMP_EPSILON);
}

TEST_F(TestSceneCommand, resource_file_rename_can_undo_and_redo) {
    auto* project_settings = gobot::Object::New<gobot::ProjectSettings>();
    const std::filesystem::path temp_root =
            std::filesystem::temp_directory_path() / "gobot_resource_rename_command_test";
    std::error_code error;
    std::filesystem::remove_all(temp_root, error);
    ASSERT_TRUE(std::filesystem::create_directories(temp_root, error));
    ASSERT_FALSE(error);

    ASSERT_TRUE(gobot::ProjectSettings::GetInstance()->SetProjectPath(temp_root.string()));

    const std::filesystem::path old_path = temp_root / "old_name.py";
    const std::filesystem::path new_path = temp_root / "new_name.py";
    {
        std::ofstream output(old_path, std::ios::out | std::ios::trunc);
        ASSERT_TRUE(output.is_open());
        output << "print('rename test')\n";
    }
    auto python_script_loader = gobot::MakeRef<gobot::ResourceFormatLoaderPythonScript>();
    gobot::ResourceLoader::AddResourceFormatLoader(python_script_loader, true);
    gobot::Ref<gobot::Resource> resource = gobot::ResourceLoader::Load("res://old_name.py", "PythonScript");
    ASSERT_TRUE(resource.IsValid());
    EXPECT_EQ(resource->GetPath(), "res://old_name.py");

    gobot::SceneCommandStack stack;
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::RenameResourceFileCommand>(
            "res://old_name.py",
            "res://new_name.py")));
    EXPECT_FALSE(stack.IsDirty());
    EXPECT_FALSE(std::filesystem::exists(old_path));
    EXPECT_TRUE(std::filesystem::exists(new_path));
    EXPECT_EQ(resource->GetPath(), "res://new_name.py");

    ASSERT_TRUE(stack.Undo());
    EXPECT_FALSE(stack.IsDirty());
    EXPECT_TRUE(std::filesystem::exists(old_path));
    EXPECT_FALSE(std::filesystem::exists(new_path));
    EXPECT_EQ(resource->GetPath(), "res://old_name.py");

    ASSERT_TRUE(stack.Redo());
    EXPECT_FALSE(stack.IsDirty());
    EXPECT_FALSE(std::filesystem::exists(old_path));
    EXPECT_TRUE(std::filesystem::exists(new_path));
    EXPECT_EQ(resource->GetPath(), "res://new_name.py");

    resource.Reset();
    gobot::ResourceLoader::RemoveResourceFormatLoader(python_script_loader);
    python_script_loader.Reset();
    std::filesystem::remove_all(temp_root, error);
    gobot::Object::Delete(project_settings);
}

TEST_F(TestSceneCommand, resource_file_rename_updates_scene_external_resource_paths) {
    auto* project_settings = gobot::Object::New<gobot::ProjectSettings>();
    const std::filesystem::path temp_root =
            std::filesystem::temp_directory_path() / "gobot_resource_rename_scene_reference_test";
    std::error_code error;
    std::filesystem::remove_all(temp_root, error);
    ASSERT_TRUE(std::filesystem::create_directories(temp_root / "scripts", error));
    ASSERT_FALSE(error);

    ASSERT_TRUE(gobot::ProjectSettings::GetInstance()->SetProjectPath(temp_root.string()));

    const std::filesystem::path old_path = temp_root / "scripts" / "cartpole.py";
    const std::filesystem::path new_path = temp_root / "scripts" / "cartpole_controller.py";
    {
        std::ofstream output(old_path, std::ios::out | std::ios::trunc);
        ASSERT_TRUE(output.is_open());
        output << "print('rename scene reference test')\n";
    }

    const std::filesystem::path scene_path = temp_root / "cartpole.jscn";
    {
        std::ofstream output(scene_path, std::ios::out | std::ios::trunc);
        ASSERT_TRUE(output.is_open());
        output << R"({
    "__EXT_RESOURCES__": [
        {
            "__ID__": "script_1",
            "__PATH__": "res://scripts/cartpole.py",
            "__TYPE__": "PythonScript"
        },
        {
            "__ID__": "scene_1",
            "__PATH__": "res://robot.jscn",
            "__TYPE__": "PackedScene"
        }
    ],
    "__META_TYPE__": "SCENE",
    "__NODES__": [],
    "__SUB_RESOURCES__": [],
    "__TYPE__": "PackedScene",
    "__VERSION__": 2
})";
    }

    gobot::SceneCommandStack stack;
    ASSERT_TRUE(stack.Execute(std::make_unique<gobot::RenameResourceFileCommand>(
            "res://scripts/cartpole.py",
            "res://scripts/cartpole_controller.py")));
    EXPECT_FALSE(std::filesystem::exists(old_path));
    EXPECT_TRUE(std::filesystem::exists(new_path));

    std::ifstream renamed_scene(scene_path);
    ASSERT_TRUE(renamed_scene.is_open());
    std::string renamed_scene_text((std::istreambuf_iterator<char>(renamed_scene)),
                                   std::istreambuf_iterator<char>());
    EXPECT_NE(renamed_scene_text.find("res://scripts/cartpole_controller.py"), std::string::npos);
    EXPECT_EQ(renamed_scene_text.find("res://scripts/cartpole.py"), std::string::npos);
    EXPECT_NE(renamed_scene_text.find("res://robot.jscn"), std::string::npos);

    ASSERT_TRUE(stack.Undo());
    EXPECT_TRUE(std::filesystem::exists(old_path));
    EXPECT_FALSE(std::filesystem::exists(new_path));

    std::ifstream restored_scene(scene_path);
    ASSERT_TRUE(restored_scene.is_open());
    std::string restored_scene_text((std::istreambuf_iterator<char>(restored_scene)),
                                    std::istreambuf_iterator<char>());
    EXPECT_NE(restored_scene_text.find("res://scripts/cartpole.py"), std::string::npos);
    EXPECT_EQ(restored_scene_text.find("res://scripts/cartpole_controller.py"), std::string::npos);

    std::filesystem::remove_all(temp_root, error);
    gobot::Object::Delete(project_settings);
}
