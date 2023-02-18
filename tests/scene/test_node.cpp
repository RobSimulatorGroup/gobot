/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-1-20
*/

#include <gtest/gtest.h>

#include <gobot/scene/node.hpp>
#include <gobot/scene/window.hpp>

class TestNode : public testing::Test {
protected:
    void SetUp() override {
        tree = gobot::SceneTree::New<gobot::SceneTree>();
        tree->Initialize();
        node = gobot::Node::New<gobot::Node>();
    }

    void TearDown() override {
        tree->Initialize();
        gobot::Node::Delete(node);
        gobot::SceneTree::Delete(tree);
    }

    gobot::SceneTree *tree{};
    gobot::Node *node{};
};

// TODO: ValidateNodeName
// TODO: ValidateChildName

TEST_F(TestNode, add_child) {
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 0);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 1);

    // Check initial node setup
    ASSERT_TRUE(node->GetName() == gobot::String());
    ASSERT_FALSE(node->IsInsideTree());
    ASSERT_EQ(node->GetParent(), nullptr);
LOG_OFF;
    ASSERT_TRUE(node->GetPath().IsEmpty());
LOG_ON;
    ASSERT_EQ(node->GetChildCount(), 0);

    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node);

    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 1);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 2);

    ASSERT_TRUE(node->GetName() != gobot::String());
    ASSERT_TRUE(node->IsInsideTree());
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot(), node->GetParent());
    ASSERT_FALSE(node->GetPath().IsEmpty());
    ASSERT_EQ(node->GetChildCount(), 0);

    gobot::Node *child = gobot::SceneTree::GetInstance()->GetRoot()->GetChild(0);
    ASSERT_EQ(child, node);
}

TEST_F(TestNode, get_node_by_path) {
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node);

    gobot::Node *child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(node->GetPath());
    ASSERT_EQ(child_by_path, node);

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(gobot::NodePath("Node"));
    ASSERT_EQ(child_by_path, nullptr);

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(gobot::NodePath("/root/Node"));
    ASSERT_EQ(child_by_path, nullptr);

    node->SetName("Node");

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(node->GetPath());
    ASSERT_EQ(child_by_path, node);

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(gobot::NodePath("Node"));
    ASSERT_EQ(child_by_path, node);

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(gobot::NodePath("/root/Node"));
    ASSERT_EQ(child_by_path, node);

    bool has_node = gobot::SceneTree::GetInstance()->GetRoot()->HasNode(gobot::NodePath("/root/Node"));
    ASSERT_TRUE(has_node);

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNode(gobot::NodePath("/root/Node"));
    ASSERT_EQ(child_by_path, node);

    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot(),
              child_by_path->FindCommonParentWith(gobot::SceneTree::GetInstance()->GetRoot()));

    gobot::NodePath path_to_root = child_by_path->GetPathTo(gobot::SceneTree::GetInstance()->GetRoot());
    ASSERT_EQ(path_to_root, gobot::NodePath("../"));
}

TEST_F(TestNode, remove_node) {
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node);

    gobot::SceneTree::GetInstance()->GetRoot()->RemoveChild(node);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 0);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 1);

    ASSERT_FALSE(node->IsInsideTree());
    ASSERT_EQ(node->GetParent(), nullptr);
LOG_OFF;
    ASSERT_TRUE(node->GetPath().IsEmpty());
LOG_ON;
}

TEST_F(TestNode, reparent_node) {
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node);
    node->Reparent(gobot::SceneTree::GetInstance()->GetRoot());

    gobot::Node *child = gobot::SceneTree::GetInstance()->GetRoot()->GetChild(0);
    ASSERT_EQ(child, node);
    ASSERT_TRUE(node->IsInsideTree());
}
