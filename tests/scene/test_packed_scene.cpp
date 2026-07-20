/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <gobot/scene/node.hpp>
#include <gobot/scene/node_3d.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/resources/packed_scene.hpp>
#include <gobot/scene/velocity_command_debug_3d.hpp>

#include <algorithm>

namespace {

const gobot::SceneState::PropertyData* FindProperty(
        const gobot::SceneState::NodeData& node,
        const std::string& name) {
    const auto property = std::find_if(
            node.properties.begin(),
            node.properties.end(),
            [&name](const gobot::SceneState::PropertyData& candidate) {
                return candidate.name == name;
            });
    return property == node.properties.end() ? nullptr : &*property;
}

} // namespace

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

TEST(TestPackedScene, semantic_label_round_trips_as_storage_property) {
    auto* root = gobot::Object::New<gobot::Node3D>();
    root->SetName("Root");
    root->SetSemanticLabel("vehicle");

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));
    const auto* root_data = packed_scene->GetState()->GetNodeData(0);
    ASSERT_NE(root_data, nullptr);
    const auto* semantic_label = FindProperty(*root_data, "semantic_label");
    ASSERT_NE(semantic_label, nullptr);
    EXPECT_EQ(semantic_label->value.convert<std::string>(), "vehicle");

    gobot::Node* restored = packed_scene->Instantiate();
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->GetSemanticLabel(), "vehicle");

    gobot::Object::Delete(root);
    gobot::Object::Delete(restored);
}

TEST(TestPackedScene, pack_motion_robot_reads_assembly_pose_without_mutating_scene) {
    auto* robot = gobot::Object::New<gobot::Robot3D>();
    robot->SetName("robot");

    auto* joint = gobot::Object::New<gobot::Joint3D>();
    joint->SetName("joint");
    joint->SetJointType(gobot::JointType::Revolute);
    joint->SetAxis(gobot::Vector3::UnitZ());
    joint->SetLowerLimit(-2.0);
    joint->SetUpperLimit(2.0);
    joint->SetJointPosition(0.4);

    auto* link = gobot::Object::New<gobot::Link3D>();
    link->SetName("link");
    link->SetPosition({1.0, 0.0, 0.0});

    robot->AddChild(joint);
    joint->AddChild(link);
    robot->SetMode(gobot::RobotMode::Motion);
    joint->SetJointPosition(0.9);

    const gobot::Affine3 joint_transform_before = joint->GetTransform();
    const gobot::Affine3 link_transform_before = link->GetTransform();
    ASSERT_FALSE(link->GetPosition().isApprox(gobot::Vector3(1.0, 0.0, 0.0), CMP_EPSILON));

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(robot));

    EXPECT_EQ(robot->GetMode(), gobot::RobotMode::Motion);
    EXPECT_TRUE(joint->IsMotionModeEnabled());
    EXPECT_NEAR(joint->GetJointPosition(), 0.9, CMP_EPSILON);
    EXPECT_TRUE(joint->GetTransform().isApprox(joint_transform_before, CMP_EPSILON));
    EXPECT_TRUE(link->GetTransform().isApprox(link_transform_before, CMP_EPSILON));

    const gobot::Ref<gobot::SceneState> state = packed_scene->GetState();
    ASSERT_EQ(state->GetNodeCount(), 3);

    const auto* robot_data = state->GetNodeData(0);
    ASSERT_NE(robot_data, nullptr);
    const auto* mode = FindProperty(*robot_data, "mode");
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(mode->value.convert<gobot::RobotMode>(), gobot::RobotMode::Motion);

    const auto* link_data = state->GetNodeData(2);
    ASSERT_NE(link_data, nullptr);
    const auto* position = FindProperty(*link_data, "position");
    ASSERT_NE(position, nullptr);
    EXPECT_TRUE(position->value.convert<gobot::Vector3>().isApprox(
            gobot::Vector3(1.0, 0.0, 0.0),
            CMP_EPSILON));

    gobot::Object::Delete(robot);
}

TEST(TestPackedScene, velocity_command_debug_packs_config_without_runtime_state) {
    auto* root = gobot::Node3D::New<gobot::Node3D>();
    root->SetName("Root");

    auto* debug = gobot::Node3D::New<gobot::VelocityCommandDebug3D>();
    debug->SetName("velocity_debug");
    debug->SetArrowScale(0.75);
    debug->SetZOffset(0.35);
    debug->SetShowYawRate(false);
    debug->SetCommandLinearVelocity({1.0, 2.0, 0.0});
    debug->SetCommandYawRate(0.4);
    debug->SetMeasuredLinearVelocity({0.5, 0.25, 0.0});
    debug->SetMeasuredYawRate(0.2);
    debug->SetPolicyLoaded(true);
    debug->SetInputFocused(true);
    debug->SetActionNorm(3.0);
    root->AddChild(debug);

    gobot::Ref<gobot::PackedScene> packed_scene = gobot::MakeRef<gobot::PackedScene>();
    ASSERT_TRUE(packed_scene->Pack(root));

    gobot::Ref<gobot::SceneState> state = packed_scene->GetState();
    ASSERT_EQ(state->GetNodeCount(), 2);
    const auto* debug_data = state->GetNodeData(1);
    ASSERT_NE(debug_data, nullptr);
    EXPECT_EQ(debug_data->type, "VelocityCommandDebug3D");

    auto has_property = [debug_data](const std::string& name) {
        return std::any_of(debug_data->properties.begin(),
                           debug_data->properties.end(),
                           [&name](const gobot::SceneState::PropertyData& property) {
                               return property.name == name;
                           });
    };

    EXPECT_TRUE(has_property("arrow_scale"));
    EXPECT_TRUE(has_property("z_offset"));
    EXPECT_TRUE(has_property("show_yaw_rate"));
    EXPECT_FALSE(has_property("command_linear_velocity"));
    EXPECT_FALSE(has_property("command_yaw_rate"));
    EXPECT_FALSE(has_property("measured_linear_velocity"));
    EXPECT_FALSE(has_property("measured_yaw_rate"));
    EXPECT_FALSE(has_property("velocity_error"));
    EXPECT_FALSE(has_property("policy_loaded"));
    EXPECT_FALSE(has_property("input_focused"));
    EXPECT_FALSE(has_property("action_norm"));

    gobot::Node* instance = packed_scene->Instantiate();
    ASSERT_NE(instance, nullptr);
    ASSERT_EQ(instance->GetChildCount(), 1);
    auto* instanced_debug = gobot::Object::PointerCastTo<gobot::VelocityCommandDebug3D>(instance->GetChild(0));
    ASSERT_NE(instanced_debug, nullptr);
    EXPECT_NEAR(instanced_debug->GetArrowScale(), 0.75, 1.0e-6);
    EXPECT_NEAR(instanced_debug->GetZOffset(), 0.35, 1.0e-6);
    EXPECT_FALSE(instanced_debug->ShouldShowYawRate());
    EXPECT_TRUE(instanced_debug->GetCommandLinearVelocity().isZero());
    EXPECT_DOUBLE_EQ(instanced_debug->GetCommandYawRate(), 0.0);
    EXPECT_TRUE(instanced_debug->GetMeasuredLinearVelocity().isZero());
    EXPECT_DOUBLE_EQ(instanced_debug->GetMeasuredYawRate(), 0.0);
    EXPECT_FALSE(instanced_debug->IsPolicyLoaded());
    EXPECT_FALSE(instanced_debug->IsInputFocused());
    EXPECT_DOUBLE_EQ(instanced_debug->GetActionNorm(), 0.0);

    gobot::Object::Delete(root);
    gobot::Object::Delete(instance);
}
