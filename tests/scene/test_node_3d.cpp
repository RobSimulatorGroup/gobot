/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-13
*/

#include <gtest/gtest.h>

#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/window.hpp>

class TestNode3D : public testing::Test {
protected:
    void SetUp() override {
        tree = gobot::SceneTree::New<gobot::SceneTree>();
        tree->Initialize();
        node_3d = gobot::Node3D::New<gobot::Node3D>();
    }

    void TearDown() override {
        tree->Finalize();
        gobot::SceneTree::Delete(tree);
    }

    gobot::SceneTree *tree{};
    gobot::Node3D *node_3d{};
};

TEST_F(TestNode3D, simple_operations) {
//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 0);
//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 1);

    gobot::SceneTree::GetInstance()->GetRoot()->AddChild(node_3d);

//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 1);
//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 2);
//
//    ASSERT_EQ(node_3d->GetParentNode3D(), nullptr);

    gobot::Vector3 p = {0.0, 0.0, 0.0};
    node_3d->SetPosition(p);
    ASSERT_EQ(node_3d->GetPosition(), p);


}