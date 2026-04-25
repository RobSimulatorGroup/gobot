/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
*/

#include <gtest/gtest.h>

#include <gobot/scene/node.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/resources/packed_scene.hpp>

TEST(TestPackedScene, pack_records_scene_tree_structure) {
    auto* root = gobot::Node3D::New<gobot::Node3D>();
    root->SetName("RobotRoot");

    auto* arm = gobot::Node3D::New<gobot::Node3D>();
    arm->SetName("Arm");
    root->AddChild(arm);

    auto* sensor = gobot::Node::New<gobot::Node>();
    sensor->SetName("Sensor");
    arm->AddChild(sensor);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    gobot::Ref<gobot::SceneState> state = packed_scene->GetState();
    ASSERT_EQ(state->GetNodeCount(), 3);

    const auto* root_data = state->GetNodeData(0);
    ASSERT_NE(root_data, nullptr);
    EXPECT_EQ(root_data->type, "Node3D");
    EXPECT_EQ(root_data->name, "RobotRoot");
    EXPECT_EQ(root_data->parent, -1);

    const auto* arm_data = state->GetNodeData(1);
    ASSERT_NE(arm_data, nullptr);
    EXPECT_EQ(arm_data->type, "Node3D");
    EXPECT_EQ(arm_data->name, "Arm");
    EXPECT_EQ(arm_data->parent, 0);

    const auto* sensor_data = state->GetNodeData(2);
    ASSERT_NE(sensor_data, nullptr);
    EXPECT_EQ(sensor_data->type, "Node");
    EXPECT_EQ(sensor_data->name, "Sensor");
    EXPECT_EQ(sensor_data->parent, 1);

    gobot::Object::Delete(root);
}

TEST(TestPackedScene, instantiate_rebuilds_scene_tree_structure) {
    auto* root = gobot::Node3D::New<gobot::Node3D>();
    root->SetName("RobotRoot");

    auto* arm = gobot::Node3D::New<gobot::Node3D>();
    arm->SetName("Arm");
    root->AddChild(arm);

    auto* sensor = gobot::Node::New<gobot::Node>();
    sensor->SetName("Sensor");
    arm->AddChild(sensor);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    gobot::Node* instance = packed_scene->Instantiate();
    ASSERT_NE(instance, nullptr);

    EXPECT_EQ(instance->GetClassStringName(), "Node3D");
    EXPECT_EQ(instance->GetName(), "RobotRoot");
    ASSERT_EQ(instance->GetChildCount(), 1);

    gobot::Node* instanced_arm = instance->GetChild(0);
    ASSERT_NE(instanced_arm, nullptr);
    EXPECT_EQ(instanced_arm->GetClassStringName(), "Node3D");
    EXPECT_EQ(instanced_arm->GetName(), "Arm");
    EXPECT_EQ(instanced_arm->GetParent(), instance);
    ASSERT_EQ(instanced_arm->GetChildCount(), 1);

    gobot::Node* instanced_sensor = instanced_arm->GetChild(0);
    ASSERT_NE(instanced_sensor, nullptr);
    EXPECT_EQ(instanced_sensor->GetClassStringName(), "Node");
    EXPECT_EQ(instanced_sensor->GetName(), "Sensor");
    EXPECT_EQ(instanced_sensor->GetParent(), instanced_arm);

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}
