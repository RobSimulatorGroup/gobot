/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-7
*/

#include <gtest/gtest.h>

#include <gobot/scene/node.hpp>
#include <gobot/scene/window.hpp>

class TestNodeComplex : public testing::Test {
protected:
    void SetUp() override {
        tree = gobot::SceneTree::New<gobot::SceneTree>();
        tree->Initialize();
        node1 = gobot::Node::New<gobot::Node>();
        node2 = gobot::Node::New<gobot::Node>();
        node1_1 = gobot::Node::New<gobot::Node>();
    }

    void TearDown() override {
        // TODO(wqq): Delete node directly is not right when node already on the scene tree
        gobot::Node::Delete(node1_1);
        gobot::Node::Delete(node1);
        gobot::Node::Delete(node2);
        gobot::SceneTree::Delete(tree);
    }

    gobot::SceneTree *tree{};
    gobot::Node *node1{};
    gobot::Node *node2{};
    gobot::Node *node1_1{};
};

TEST_F(TestNodeComplex, add_child) {
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node1);
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node2);

    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot(), node1->GetParent());
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot(), node2->GetParent());

    node1->AddChild(node1_1);

    ASSERT_TRUE(node1_1->IsInsideTree());
    ASSERT_EQ(node1_1->GetParent(), node1);
    ASSERT_EQ(node1->GetChildCount(), 1);

    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 2);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 4);

    // Nodes should be accessible via GetChild(...)
    gobot::Node *child1 = gobot::SceneTree::GetInstance()->GetRoot()->GetChild(0);
    ASSERT_EQ(child1, node1);

    gobot::Node *child2 = gobot::SceneTree::GetInstance()->GetRoot()->GetChild(1);
    ASSERT_EQ(child2, node2);

    gobot::Node *child1_1 = node1->GetChild(0);
    ASSERT_EQ(child1_1, node1_1);

    ASSERT_TRUE(gobot::SceneTree::GetInstance()->GetRoot()->IsAncestorOf(node1_1));

    gobot::NodePath path_to_root = child1_1->GetPathTo(gobot::SceneTree::GetInstance()->GetRoot());
    ASSERT_EQ(path_to_root, gobot::NodePath("../.."));

    // Removed children nodes should not affect their parent in the scene tree
    node1->RemoveChild(node1_1);

    ASSERT_EQ(node1_1->GetParent(), nullptr);
    ASSERT_EQ(node1->GetChildCount(), 0);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 3);
}

TEST_F(TestNodeComplex, modify_node) {
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node1);
    node1->AddSibling(node2);
    node1->AddChild(node1_1);

    // Node should be in the expected order when reparented
    ASSERT_EQ(node2->GetChildCount(), 0);

    node1_1->Reparent(node2);
    ASSERT_EQ(node1->GetChildCount(), 0);
    ASSERT_EQ(node2->GetChildCount(), 1);
    ASSERT_EQ(node1_1->GetParent(), node2);

    gobot::Node *child = node2->GetChild(0);
    ASSERT_EQ(child, node1_1);

    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 2);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 4);
}

TEST_F(TestNodeComplex, get_node) {
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node1);
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node2);
    node1->AddChild(node1_1);

    gobot::Node *child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(node1->GetPath());
    ASSERT_EQ(child_by_path, node1);

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(node2->GetPath());
    ASSERT_EQ(child_by_path, node2);

    child_by_path = gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(node1_1->GetPath());
    ASSERT_EQ(child_by_path, node1_1);

    node1->SetName("Node1");
    node1_1->SetName("NestedNode");

    child_by_path = node1->GetNodeOrNull(gobot::NodePath("NestedNode"));
    ASSERT_EQ(child_by_path, node1_1);

    child_by_path =
            gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(gobot::NodePath("/root/Node1/NestedNode"));
    ASSERT_EQ(child_by_path, node1_1);

    child_by_path =
            gobot::SceneTree::GetInstance()->GetRoot()->GetNodeOrNull(gobot::NodePath("Node1/NestedNode"));
    ASSERT_EQ(child_by_path, node1_1);
}

TEST_F(TestNodeComplex, add_sibling) {
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node1);
    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node2);
    node1->AddChild(node1_1);

    // Nodes added as siblings of another node should be right next to it
    node1->RemoveChild(node1_1);
    node1->AddSibling(node1_1);

    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 3);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 4);

    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChild(0), node1);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChild(1), node1_1);
    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChild(2), node2);

    // Nodes name should be validated as unique and renamed if not
    node1->SetName("Node9");
    node1_1->SetName("Node9");
    ASSERT_EQ(node1_1->GetName(), "Node10");
}
