#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include <gobot/rendering/render_server.hpp>
#include <gobot/scene/mesh_instance_3d.hpp>
#include <gobot/scene/node_creation_registry.hpp>
#include <gobot/scene/resources/primitive_mesh.hpp>

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
    const auto* mesh_instance = FindEntry("MeshInstance3D");
    const auto* box_mesh = FindEntry("BoxMeshInstance3D");

    ASSERT_NE(node, nullptr);
    ASSERT_NE(node_3d, nullptr);
    ASSERT_NE(mesh_instance, nullptr);
    ASSERT_NE(box_mesh, nullptr);

    EXPECT_TRUE(node->parent_id.empty());
    EXPECT_EQ(node_3d->parent_id, "Node");
    EXPECT_EQ(mesh_instance->parent_id, "Node3D");
    EXPECT_EQ(box_mesh->parent_id, "MeshInstance3D");
}

TEST(TestNodeCreationRegistry, creates_box_mesh_shortcut) {
    auto render_server = std::make_unique<gobot::RenderServer>();
    gobot::Node* node = gobot::NodeCreationRegistry::CreateNode("BoxMeshInstance3D");
    ASSERT_NE(node, nullptr);

    auto* mesh_instance = gobot::Object::PointerCastTo<gobot::MeshInstance3D>(node);
    ASSERT_NE(mesh_instance, nullptr);
    EXPECT_EQ(mesh_instance->GetName(), "Box");
    ASSERT_TRUE(mesh_instance->GetMesh().IsValid());
    EXPECT_NE(mesh_instance->GetMesh().DynamicPointerCast<gobot::BoxMesh>().Get(), nullptr);

    gobot::Object::Delete(node);
}
