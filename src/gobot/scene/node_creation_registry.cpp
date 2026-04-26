/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/node_creation_registry.hpp"

#include <algorithm>

#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"

namespace gobot {

namespace {

std::vector<NodeCreationEntry>& MutableNodeTypes() {
    static std::vector<NodeCreationEntry> node_types;
    return node_types;
}

bool& BuiltInsRegistered() {
    static bool registered = false;
    return registered;
}

template <typename T>
Node* CreateNodeInstance() {
    return Object::New<T>();
}

} // namespace

bool NodeCreationRegistry::RegisterNodeType(NodeCreationEntry entry) {
    if (entry.id.empty() || !entry.create) {
        return false;
    }

    auto& node_types = MutableNodeTypes();
    const auto existing = std::find_if(node_types.begin(), node_types.end(),
                                       [&entry](const NodeCreationEntry& item) {
                                           return item.id == entry.id;
                                       });
    if (existing != node_types.end()) {
        *existing = std::move(entry);
        return true;
    }

    node_types.push_back(std::move(entry));
    return true;
}

const std::vector<NodeCreationEntry>& NodeCreationRegistry::GetNodeTypes() {
    EnsureBuiltInNodeTypesRegistered();
    return MutableNodeTypes();
}

const NodeCreationEntry* NodeCreationRegistry::FindNodeType(const std::string& id) {
    const auto& node_types = GetNodeTypes();
    const auto iter = std::find_if(node_types.begin(), node_types.end(),
                                   [&id](const NodeCreationEntry& item) {
                                       return item.id == id;
                                   });
    return iter == node_types.end() ? nullptr : &(*iter);
}

Node* NodeCreationRegistry::CreateNode(const std::string& id) {
    const auto* entry = FindNodeType(id);
    return entry == nullptr ? nullptr : entry->create();
}

void NodeCreationRegistry::EnsureBuiltInNodeTypesRegistered() {
    if (BuiltInsRegistered()) {
        return;
    }

    BuiltInsRegistered() = true;

    RegisterNodeType({
        "Node",
        "Node",
        "",
        "Base scene node for hierarchy and lifecycle.",
        []() -> Node* { return CreateNodeInstance<Node>(); }
    });

    RegisterNodeType({
        "Node3D",
        "Node3D",
        "Node",
        "3D scene node with local transform, global transform, and visibility.",
        []() -> Node* { return CreateNodeInstance<Node3D>(); }
    });

    RegisterNodeType({
        "MeshInstance3D",
        "MeshInstance3D",
        "Node3D",
        "3D node that renders a mesh resource.",
        []() -> Node* { return CreateNodeInstance<MeshInstance3D>(); }
    });

    RegisterNodeType({
        "CollisionShape3D",
        "CollisionShape3D",
        "Node3D",
        "3D node that provides collision geometry for physics and collision queries.",
        []() -> Node* { return CreateNodeInstance<CollisionShape3D>(); }
    });

    RegisterNodeType({
        "BoxMeshInstance3D",
        "Box Mesh",
        "MeshInstance3D",
        "Creates a MeshInstance3D with a default BoxMesh resource.",
        []() -> Node* {
            auto* node = Object::New<MeshInstance3D>();
            node->SetName("Box");
            node->SetMesh(MakeRef<BoxMesh>());
            return node;
        }
    });
}

} // namespace gobot
