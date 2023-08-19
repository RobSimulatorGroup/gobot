/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-2-13
*/

#include <gtest/gtest.h>

#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/window.hpp>
#include <gobot/rendering/render_server.hpp>

class TestNode3D : public testing::Test {
protected:
    void SetUp() override {
        render_server = std::make_unique<gobot::RenderServer>();

        tree = gobot::SceneTree::New<gobot::SceneTree>();
        tree->Initialize();
        node1 = gobot::Node3D::New<gobot::Node3D>();
        node1_1 = gobot::Node3D::New<gobot::Node3D>();
    }

    void TearDown() override {
        tree->Finalize();
        gobot::SceneTree::Delete(tree);
    }

    gobot::SceneTree *tree{};
    gobot::Node3D *node1{};
    gobot::Node3D *node1_1{};
    std::unique_ptr<gobot::RenderServer> render_server;
};

TEST_F(TestNode3D, simple_operations) {
    using namespace gobot;

    SceneTree::GetInstance()->GetRoot()->AddChild(node1);

    Vector3 p = {1.0, 1.0, 1.0};
    node1->SetPosition(p);
    ASSERT_EQ(node1->GetPosition(), p);

    node1->SetRotationEditMode(Node3D::RotationEditMode::Euler);
    ASSERT_EQ(node1->GetRotationEditMode(), Node3D::RotationEditMode::Euler);

    node1->SetEulerOrder(EulerOrder::SZYX);
    ASSERT_EQ(node1->GetEulerOrder(), EulerOrder::SZYX);

    auto euler_radian = EulerAngle{Math_PI * 0.25, -Math_PI * 0.15, Math_PI * 0.1};
    node1->SetEuler(euler_radian);
    ASSERT_FLOAT_EQ(node1->GetEuler().x(), Math_PI * 0.25);
    ASSERT_FLOAT_EQ(node1->GetEuler().y(), -Math_PI * 0.15);
    ASSERT_FLOAT_EQ(node1->GetEuler().z(), Math_PI * 0.1);

    auto euler_degree = EulerAngle{30.0, 60.0, 90.0};
    node1->SetEulerDegree(euler_degree);
    ASSERT_FLOAT_EQ(node1->GetEulerDegree().x(), 30.0);
    ASSERT_FLOAT_EQ(node1->GetEulerDegree().y(), 60.0);
    ASSERT_FLOAT_EQ(node1->GetEulerDegree().z(), 90.0);

    auto scale = Vector3{1.0, 2.0, 3.0};
    node1->SetScale(scale);
    ASSERT_FLOAT_EQ(node1->GetScale().x(), 1.0);
    ASSERT_FLOAT_EQ(node1->GetScale().y(), 2.0);
    ASSERT_FLOAT_EQ(node1->GetScale().z(), 3.0);

    node1->SetTransform(Affine3::Identity());
    ASSERT_TRUE(node1->GetTransform().isApprox(Affine3::Identity(), CMP_EPSILON));

    node1->SetQuaternion(Quaternion::Identity());
    ASSERT_TRUE(node1->GetQuaternion().isApprox(Quaternion::Identity(), CMP_EPSILON));

    node1->SetGlobalTransform(Affine3::Identity());
    node1->GetGlobalTransform();
    ASSERT_TRUE(node1->GetGlobalTransform().isApprox(Affine3::Identity(), CMP_EPSILON));

    ASSERT_EQ(SceneTree::GetInstance()->GetRoot()->GetChild(0), node1);

    node1->SetGlobalPosition(p * 2);
    ASSERT_TRUE(node1->GetGlobalPosition().isApprox(p * 2, CMP_EPSILON));
}

TEST_F(TestNode3D, complex_operations) {
    using namespace gobot;

    SceneTree::GetInstance()->GetRoot()->AddChild(node1);
    node1->SetName("base");

    node1->SetTransform(Affine3::Identity());

    ASSERT_TRUE(node1->GetTransform().isApprox(Affine3::Identity(), CMP_EPSILON));
    ASSERT_TRUE(node1->GetGlobalTransform().isApprox(Affine3::Identity(), CMP_EPSILON));

    auto euler1 = EulerAngle{0.0, 0.0, Math_PI * 0.25};
    node1->SetEuler(euler1);
    auto position1 = Vector3{1.0, 1.0, 1.0};
    node1->SetPosition(position1);

    Affine3 test{Affine3::Identity()};
    test.SetEulerAngle(euler1, node1->GetEulerOrder());
    test.translation() = position1;
    ASSERT_TRUE(node1->GetTransform().isApprox(test, CMP_EPSILON));
    ASSERT_TRUE(node1->GetGlobalTransform().isApprox(test, CMP_EPSILON));

    node1->AddChild(node1_1);
    node1_1->SetName("link");

    node1->RotateZ(Math_PI * 0.25);

    // link should rotate with base transform
    Affine3 m{Affine3::Identity()};
    m.linear() = AngleAxis(Math_PI * 0.5, Vector3::UnitZ()).toRotationMatrix();
    m.translation() = position1;
    ASSERT_TRUE(node1_1->GetTransform().isApprox(Affine3::Identity(), CMP_EPSILON));
    ASSERT_TRUE(node1_1->GetGlobalTransform().isApprox(m, CMP_EPSILON));
}