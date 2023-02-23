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

// todo: UpdateLocalTransform
// todo: UpdateEulerAndScale

TEST_F(TestNode3D, simple_operations) {
    using namespace gobot;
//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 0);
//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 1);

    SceneTree::GetInstance()->GetRoot()->AddChild(node_3d);

//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetRoot()->GetChildCount(), 1);
//    ASSERT_EQ(gobot::SceneTree::GetInstance()->GetNodeCount(), 2);
//
//    ASSERT_EQ(node_3d->GetParentNode3D(), nullptr);

    Vector3 p = {1.0, 1.0, 1.0};
    node_3d->SetPosition(p);
    ASSERT_EQ(node_3d->GetPosition(), p);

    node_3d->SetRotationEditMode(Node3D::RotationEditMode::Euler);
    ASSERT_EQ(node_3d->GetRotationEditMode(), Node3D::RotationEditMode::Euler);

    node_3d->SetEulerOrder(EulerOrder::SZYX);
    ASSERT_EQ(node_3d->GetEulerOrder(), EulerOrder::SZYX);

    auto euler_radian = EulerAngle{Math_PI * 0.25, -Math_PI * 0.15, Math_PI * 0.1};
    node_3d->SetEuler(euler_radian);
    ASSERT_FLOAT_EQ(node_3d->GetEuler().x(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(node_3d->GetEuler().y(), -Math_PI * 0.15);
    ASSERT_FLOAT_EQ(node_3d->GetEuler().z(), Math_PI * 0.1);

    auto euler_degree = EulerAngle{30.0, 60.0, 90.0};
    node_3d->SetEulerDegree(euler_degree);
    ASSERT_FLOAT_EQ(node_3d->GetEulerDegree().x(), 30.0);
    ASSERT_FLOAT_EQ(node_3d->GetEulerDegree().y(), 60.0);
    ASSERT_FLOAT_EQ(node_3d->GetEulerDegree().z(), 90.0);

    // todo: transform change is propagated (not implemented) and not updated at once
    auto scale = Vector3{1.0, 2.0, 3.0};
    node_3d->SetScale(scale);
    ASSERT_FLOAT_EQ(node_3d->GetScale().x(), 1.0);
    ASSERT_FLOAT_EQ(node_3d->GetScale().y(), 2.0);
    ASSERT_FLOAT_EQ(node_3d->GetScale().z(), 3.0);

    node_3d->SetTransform(Affine3::Identity());
    ASSERT_TRUE(node_3d->GetTransform().isApprox(Affine3::Identity(), CMP_EPSILON));

    node_3d->SetQuaternion(Quaternion::Identity());
    ASSERT_TRUE(node_3d->GetQuaternion().isApprox(Quaternion::Identity(), CMP_EPSILON));

    node_3d->SetGlobalTransform(Affine3::Identity());
    ASSERT_TRUE(node_3d->GetGlobalTransform().isApprox(Affine3::Identity(), CMP_EPSILON));

    node_3d->SetGlobalPosition(p * 2);
    ASSERT_TRUE(node_3d->GetGlobalPosition().isApprox(p * 2, CMP_EPSILON));
}
