/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/node_creation_registry.hpp"

#include <algorithm>
#include <utility>

#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/link_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/robot_3d.hpp"

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

template <typename T>
Node* CreateMeshInstanceWithMesh(std::string name) {
    auto* node = Object::New<MeshInstance3D>();
    node->SetName(std::move(name));
    node->SetMesh(MakeRef<T>());
    return node;
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
        "Robot3D",
        "Robot3D",
        "Node3D",
        "Root node for an imported or authored robot model.",
        []() -> Node* { return CreateNodeInstance<Robot3D>(); }
    });

    RegisterNodeType({
        "Link3D",
        "Link3D",
        "Node3D",
        "Robot link node with inertial metadata and visual/collision children.",
        []() -> Node* { return CreateNodeInstance<Link3D>(); }
    });

    RegisterNodeType({
        "Joint3D",
        "Joint3D",
        "Node3D",
        "Robot joint node that connects parent and child links.",
        []() -> Node* { return CreateNodeInstance<Joint3D>(); }
    });

    RegisterNodeType({
        "BoxMeshInstance3D",
        "Box Mesh",
        "MeshInstance3D",
        "MeshInstance3D preset with a BoxMesh resource assigned.",
        []() -> Node* { return CreateMeshInstanceWithMesh<BoxMesh>("Box"); }
    });

    RegisterNodeType({
        "CylinderMeshInstance3D",
        "Cylinder Mesh",
        "MeshInstance3D",
        "MeshInstance3D preset with a CylinderMesh resource assigned.",
        []() -> Node* { return CreateMeshInstanceWithMesh<CylinderMesh>("Cylinder"); }
    });

    RegisterNodeType({
        "SphereMeshInstance3D",
        "Sphere Mesh",
        "MeshInstance3D",
        "MeshInstance3D preset with a SphereMesh resource assigned.",
        []() -> Node* { return CreateMeshInstanceWithMesh<SphereMesh>("Sphere"); }
    });

}

} // namespace gobot
