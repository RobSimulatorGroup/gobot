#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/environment_3d.hpp>
#include <gobot/scene/light_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node_creation_registry.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>
#include <gobot/scene/robot_3d.hpp>
#include <gobot/scene/sensor_3d.hpp>
#include <gobot/scene/terrain_3d.hpp>
#include <gobot/scene/velocity_command_debug_3d.hpp>

namespace {

const gobot::NodeCreationEntry* FindEntry(const std::string& id) {
    const auto& entries = gobot::NodeCreationRegistry::GetNodeTypes();
    const auto iter = std::find_if(entries.begin(), entries.end(),
                                   [&id](const gobot::NodeCreationEntry& entry) {
                                       return entry.id == id;
                                   });
    return iter == entries.end() ? nullptr : &(*iter);
}

}

TEST(TestNodeCreationRegistry, built_in_entries_keep_node_inheritance_shape) {
    const auto* node = FindEntry("Node");
    const auto* node_3d = FindEntry("Node3D");
    const auto* collision_shape = FindEntry("CollisionShape3D");
    const auto* terrain = FindEntry("Terrain3D");
    const auto* robot = FindEntry("Robot3D");
    const auto* link = FindEntry("Link3D");
    const auto* joint = FindEntry("Joint3D");
    const auto* sensor = FindEntry("Sensor3D");
    const auto* imu_sensor = FindEntry("IMUSensor3D");
    const auto* angular_momentum_sensor = FindEntry("AngularMomentumSensor3D");
    const auto* contact_sensor = FindEntry("ContactSensor3D");
    const auto* raycast_sensor = FindEntry("RayCastSensor3D");
    const auto* terrain_height_sensor = FindEntry("TerrainHeightSensor3D");
    const auto* height_scanner_sensor = FindEntry("HeightScanner3D");
    const auto* velocity_command_debug = FindEntry("VelocityCommandDebug3D");
    const auto* mesh_instance = FindEntry("MeshInstance3D");
    const auto* environment = FindEntry("Environment3D");
    const auto* directional_light = FindEntry("DirectionalLight3D");
    const auto* point_light = FindEntry("PointLight3D");
    const auto* spot_light = FindEntry("SpotLight3D");
    const auto* box_mesh = FindEntry("BoxMeshInstance3D");
    const auto* cylinder_mesh = FindEntry("CylinderMeshInstance3D");
    const auto* sphere_mesh = FindEntry("SphereMeshInstance3D");

    ASSERT_NE(node, nullptr);
    ASSERT_NE(node_3d, nullptr);
    ASSERT_NE(collision_shape, nullptr);
    ASSERT_NE(terrain, nullptr);
    ASSERT_NE(robot, nullptr);
    ASSERT_NE(link, nullptr);
    ASSERT_NE(joint, nullptr);
    ASSERT_NE(sensor, nullptr);
    ASSERT_NE(imu_sensor, nullptr);
    ASSERT_NE(angular_momentum_sensor, nullptr);
    ASSERT_NE(contact_sensor, nullptr);
    ASSERT_NE(raycast_sensor, nullptr);
    ASSERT_NE(terrain_height_sensor, nullptr);
    ASSERT_NE(height_scanner_sensor, nullptr);
    ASSERT_NE(velocity_command_debug, nullptr);
    ASSERT_NE(mesh_instance, nullptr);
    ASSERT_NE(environment, nullptr);
    ASSERT_NE(directional_light, nullptr);
    ASSERT_NE(point_light, nullptr);
    ASSERT_NE(spot_light, nullptr);
    ASSERT_NE(box_mesh, nullptr);
    ASSERT_NE(cylinder_mesh, nullptr);
    ASSERT_NE(sphere_mesh, nullptr);

    EXPECT_TRUE(node->parent_id.empty());
    EXPECT_EQ(node_3d->parent_id, "Node");
    EXPECT_EQ(collision_shape->parent_id, "Node3D");
    EXPECT_EQ(terrain->parent_id, "Node3D");
    EXPECT_EQ(robot->parent_id, "Node3D");
    EXPECT_EQ(link->parent_id, "Node3D");
    EXPECT_EQ(joint->parent_id, "Node3D");
    EXPECT_EQ(sensor->parent_id, "Node3D");
    EXPECT_EQ(imu_sensor->parent_id, "Sensor3D");
    EXPECT_EQ(angular_momentum_sensor->parent_id, "Sensor3D");
    EXPECT_EQ(contact_sensor->parent_id, "Sensor3D");
    EXPECT_EQ(raycast_sensor->parent_id, "Sensor3D");
    EXPECT_EQ(terrain_height_sensor->parent_id, "RayCastSensor3D");
    EXPECT_EQ(height_scanner_sensor->parent_id, "TerrainHeightSensor3D");
    EXPECT_EQ(velocity_command_debug->parent_id, "Node3D");
    EXPECT_EQ(mesh_instance->parent_id, "Node3D");
    EXPECT_EQ(environment->parent_id, "Node3D");
    EXPECT_EQ(directional_light->parent_id, "Light3D");
    EXPECT_EQ(point_light->parent_id, "Light3D");
    EXPECT_EQ(spot_light->parent_id, "PointLight3D");
    EXPECT_EQ(box_mesh->parent_id, "MeshInstance3D");
    EXPECT_EQ(cylinder_mesh->parent_id, "MeshInstance3D");
    EXPECT_EQ(sphere_mesh->parent_id, "MeshInstance3D");
}

TEST(TestNodeCreationRegistry, creates_velocity_command_debug_node) {
    gobot::Node* node = gobot::NodeCreationRegistry::CreateNode("VelocityCommandDebug3D");
    ASSERT_NE(node, nullptr);

    auto* debug_node = gobot::Object::PointerCastTo<gobot::VelocityCommandDebug3D>(node);
    ASSERT_NE(debug_node, nullptr);
    EXPECT_TRUE(debug_node->IsEnabled());
    EXPECT_TRUE(debug_node->ShouldShowCommandVelocity());
    EXPECT_TRUE(debug_node->ShouldShowMeasuredVelocity());
    EXPECT_TRUE(debug_node->ShouldShowYawRate());
    EXPECT_NEAR(debug_node->GetArrowScale(), 0.55, 1.0e-6);
    EXPECT_NEAR(debug_node->GetZOffset(), 0.30, 1.0e-6);

    gobot::Object::Delete(node);
}

TEST(TestNodeCreationRegistry, creates_robot_semantic_nodes) {
    gobot::Node* robot_node = gobot::NodeCreationRegistry::CreateNode("Robot3D");
    ASSERT_NE(robot_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::Robot3D>(robot_node), nullptr);
    gobot::Object::Delete(robot_node);

    gobot::Node* link_node = gobot::NodeCreationRegistry::CreateNode("Link3D");
    ASSERT_NE(link_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::Link3D>(link_node), nullptr);
    gobot::Object::Delete(link_node);

    gobot::Node* joint_node = gobot::NodeCreationRegistry::CreateNode("Joint3D");
    ASSERT_NE(joint_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::Joint3D>(joint_node), nullptr);
    gobot::Object::Delete(joint_node);
}

TEST(TestNodeCreationRegistry, creates_sensor_nodes) {
    gobot::Node* imu_node = gobot::NodeCreationRegistry::CreateNode("IMUSensor3D");
    ASSERT_NE(imu_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::IMUSensor3D>(imu_node), nullptr);
    gobot::Object::Delete(imu_node);

    gobot::Node* contact_node = gobot::NodeCreationRegistry::CreateNode("ContactSensor3D");
    ASSERT_NE(contact_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::ContactSensor3D>(contact_node), nullptr);
    gobot::Object::Delete(contact_node);

    gobot::Node* angular_momentum_node = gobot::NodeCreationRegistry::CreateNode("AngularMomentumSensor3D");
    ASSERT_NE(angular_momentum_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::AngularMomentumSensor3D>(angular_momentum_node), nullptr);
    gobot::Object::Delete(angular_momentum_node);

    gobot::Node* terrain_height_node = gobot::NodeCreationRegistry::CreateNode("HeightScanner3D");
    ASSERT_NE(terrain_height_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::HeightScanner3D>(terrain_height_node), nullptr);
    gobot::Object::Delete(terrain_height_node);

    gobot::Node* raycast_node = gobot::NodeCreationRegistry::CreateNode("RayCastSensor3D");
    ASSERT_NE(raycast_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::RayCastSensor3D>(raycast_node), nullptr);
    gobot::Object::Delete(raycast_node);

    gobot::Node* terrain_height_sensor_node = gobot::NodeCreationRegistry::CreateNode("TerrainHeightSensor3D");
    ASSERT_NE(terrain_height_sensor_node, nullptr);
    EXPECT_NE(gobot::Object::PointerCastTo<gobot::TerrainHeightSensor3D>(terrain_height_sensor_node), nullptr);
    gobot::Object::Delete(terrain_height_sensor_node);
}

TEST(TestNodeCreationRegistry, creates_mesh_instance_node) {
    gobot::Node* node = gobot::NodeCreationRegistry::CreateNode("MeshInstance3D");
    ASSERT_NE(node, nullptr);

    auto* mesh_instance = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(node);
    ASSERT_NE(mesh_instance, nullptr);
    EXPECT_FALSE(mesh_instance->GetMesh().IsValid());

    gobot::Object::Delete(node);
}

TEST(TestNodeCreationRegistry, creates_terrain_node) {
    gobot::Node* node = gobot::NodeCreationRegistry::CreateNode("Terrain3D");
    ASSERT_NE(node, nullptr);

    auto* terrain = gobot::Object::PointerCastTo<gobot::Terrain3D>(node);
    ASSERT_NE(terrain, nullptr);
    EXPECT_EQ(terrain->GetBoxes().size(), 0);
    EXPECT_EQ(terrain->GetHeightFields().size(), 0);

    gobot::Object::Delete(node);
}

TEST(TestNodeCreationRegistry, creates_mesh_instance_presets_with_mesh_resources) {
    auto render_server = std::make_unique<gobot::RenderServer>();

    gobot::Node* cylinder_node = gobot::NodeCreationRegistry::CreateNode("CylinderMeshInstance3D");
    ASSERT_NE(cylinder_node, nullptr);
    auto* cylinder_instance = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(cylinder_node);
    ASSERT_NE(cylinder_instance, nullptr);
    EXPECT_EQ(cylinder_instance->GetName(), "Cylinder");
    EXPECT_TRUE(gobot::dynamic_pointer_cast<gobot::CylinderMesh>(cylinder_instance->GetMesh()).IsValid());
    gobot::Object::Delete(cylinder_node);

    gobot::Node* sphere_node = gobot::NodeCreationRegistry::CreateNode("SphereMeshInstance3D");
    ASSERT_NE(sphere_node, nullptr);
    auto* sphere_instance = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(sphere_node);
    ASSERT_NE(sphere_instance, nullptr);
    EXPECT_EQ(sphere_instance->GetName(), "Sphere");
    EXPECT_TRUE(gobot::dynamic_pointer_cast<gobot::SphereMesh>(sphere_instance->GetMesh()).IsValid());
    gobot::Object::Delete(sphere_node);
}
