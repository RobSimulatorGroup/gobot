#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/joint_3d.hpp>
#include <gobot/scene/link_3d.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node_creation_registry.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>
#include <gobot/scene/robot_3d.hpp>

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
    const auto* robot = FindEntry("Robot3D");
    const auto* link = FindEntry("Link3D");
    const auto* joint = FindEntry("Joint3D");
    const auto* mesh_instance = FindEntry("MeshInstance3D");
    const auto* box_mesh = FindEntry("BoxMeshInstance3D");
    const auto* cylinder_mesh = FindEntry("CylinderMeshInstance3D");
    const auto* sphere_mesh = FindEntry("SphereMeshInstance3D");

    ASSERT_NE(node, nullptr);
    ASSERT_NE(node_3d, nullptr);
    ASSERT_NE(collision_shape, nullptr);
    ASSERT_NE(robot, nullptr);
    ASSERT_NE(link, nullptr);
    ASSERT_NE(joint, nullptr);
    ASSERT_NE(mesh_instance, nullptr);
    ASSERT_NE(box_mesh, nullptr);
    ASSERT_NE(cylinder_mesh, nullptr);
    ASSERT_NE(sphere_mesh, nullptr);

    EXPECT_TRUE(node->parent_id.empty());
    EXPECT_EQ(node_3d->parent_id, "Node");
    EXPECT_EQ(collision_shape->parent_id, "Node3D");
    EXPECT_EQ(robot->parent_id, "Node3D");
    EXPECT_EQ(link->parent_id, "Node3D");
    EXPECT_EQ(joint->parent_id, "Node3D");
    EXPECT_EQ(mesh_instance->parent_id, "Node3D");
    EXPECT_EQ(box_mesh->parent_id, "MeshInstance3D");
    EXPECT_EQ(cylinder_mesh->parent_id, "MeshInstance3D");
    EXPECT_EQ(sphere_mesh->parent_id, "MeshInstance3D");
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

TEST(TestNodeCreationRegistry, creates_mesh_instance_node) {
    gobot::Node* node = gobot::NodeCreationRegistry::CreateNode("MeshInstance3D");
    ASSERT_NE(node, nullptr);

    auto* mesh_instance = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(node);
    ASSERT_NE(mesh_instance, nullptr);
    EXPECT_FALSE(mesh_instance->GetMesh().IsValid());

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
