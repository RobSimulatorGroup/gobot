#include <gtest/gtest.h>

#include <algorithm>

#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node_creation_registry.hpp>

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
    const auto* mesh_instance = FindEntry("MeshInstance3D");

    ASSERT_NE(node, nullptr);
    ASSERT_NE(node_3d, nullptr);
    ASSERT_NE(collision_shape, nullptr);
    ASSERT_NE(mesh_instance, nullptr);

    EXPECT_TRUE(node->parent_id.empty());
    EXPECT_EQ(node_3d->parent_id, "Node");
    EXPECT_EQ(collision_shape->parent_id, "Node3D");
    EXPECT_EQ(mesh_instance->parent_id, "Node3D");
}

TEST(TestNodeCreationRegistry, creates_mesh_instance_node) {
    gobot::Node* node = gobot::NodeCreationRegistry::CreateNode("MeshInstance3D");
    ASSERT_NE(node, nullptr);

    auto* mesh_instance = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(node);
    ASSERT_NE(mesh_instance, nullptr);
    EXPECT_FALSE(mesh_instance->GetMesh().IsValid());

    gobot::Object::Delete(node);
}
