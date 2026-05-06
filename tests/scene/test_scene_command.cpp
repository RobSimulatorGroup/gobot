#include <gtest/gtest.h>

#include "gobot/core/object.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/scene_command.hpp"

namespace {

class TestSceneCommand : public ::testing::Test {
protected:
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
