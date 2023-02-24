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
        node = gobot::Node3D::New<gobot::Node3D>();
    }

    void TearDown() override {
        tree->Finalize();
        gobot::SceneTree::Delete(tree);
    }

    gobot::SceneTree *tree{};
    gobot::Node3D *node{};
};

TEST_F(TestNode3D, simple_operations) {
    using namespace gobot;

    SceneTree::GetInstance()->GetRoot()->AddChild(node);

    Vector3 p = {1.0, 1.0, 1.0};
    node->SetPosition(p);
    ASSERT_EQ(node->GetPosition(), p);

    node->SetRotationEditMode(Node3D::RotationEditMode::Euler);
    ASSERT_EQ(node->GetRotationEditMode(), Node3D::RotationEditMode::Euler);

    node->SetEulerOrder(EulerOrder::SZYX);
    ASSERT_EQ(node->GetEulerOrder(), EulerOrder::SZYX);

    auto euler_radian = EulerAngle{Math_PI * 0.25, -Math_PI * 0.15, Math_PI * 0.1};
    node->SetEuler(euler_radian);
    ASSERT_FLOAT_EQ(node->GetEuler().x(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(node->GetEuler().y(), -Math_PI * 0.15);
    ASSERT_FLOAT_EQ(node->GetEuler().z(), Math_PI * 0.1);

    auto euler_degree = EulerAngle{30.0, 60.0, 90.0};
    node->SetEulerDegree(euler_degree);
    ASSERT_FLOAT_EQ(node->GetEulerDegree().x(), 30.0);
    ASSERT_FLOAT_EQ(node->GetEulerDegree().y(), 60.0);
    ASSERT_FLOAT_EQ(node->GetEulerDegree().z(), 90.0);

    auto scale = Vector3{1.0, 2.0, 3.0};
    node->SetScale(scale);
    ASSERT_FLOAT_EQ(node->GetScale().x(), 1.0);
    ASSERT_FLOAT_EQ(node->GetScale().y(), 2.0);
    ASSERT_FLOAT_EQ(node->GetScale().z(), 3.0);

    node->SetTransform(Affine3::Identity());
    ASSERT_TRUE(node->GetTransform().isApprox(Affine3::Identity(), CMP_EPSILON));

    node->SetQuaternion(Quaternion::Identity());
    ASSERT_TRUE(node->GetQuaternion().isApprox(Quaternion::Identity(), CMP_EPSILON));

    node->SetGlobalTransform(Affine3::Identity());
    node->GetGlobalTransform();
    ASSERT_TRUE(node->GetGlobalTransform().isApprox(Affine3::Identity(), CMP_EPSILON));

    ASSERT_EQ(SceneTree::GetInstance()->GetRoot()->GetChild(0), node);

    node->SetGlobalPosition(p * 2);
    ASSERT_TRUE(node->GetGlobalPosition().isApprox(p * 2, CMP_EPSILON));
}
