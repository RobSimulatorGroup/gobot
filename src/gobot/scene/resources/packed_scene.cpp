/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/packed_scene.hpp"

#include "gobot/core/object.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/joint_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"

namespace gobot {
namespace {

int GetPropertyRestorePriority(const std::string& node_type, const std::string& property_name) {
    if (node_type == "MeshInstance3D") {
        if (property_name == "mesh") {
            return 0;
        }
        if (property_name == "mesh_material") {
            return 10;
        }
    }

    if (node_type == "Joint3D") {
        if (property_name == "position" || property_name == "rotation_degrees" || property_name == "scale") {
            return 0;
        }
        if (property_name == "joint_type") {
            return 10;
        }
        if (property_name == "parent_link" || property_name == "child_link" || property_name == "axis") {
            return 20;
        }
        if (property_name == "lower_limit" || property_name == "upper_limit" ||
            property_name == "effort_limit" || property_name == "velocity_limit") {
            return 30;
        }
        if (property_name == "joint_position") {
            return 40;
        }
    }

    return 20;
}

std::vector<SceneState::PropertyData> GetPropertiesInRestoreOrder(const SceneState::NodeData* node_data) {
    std::vector<SceneState::PropertyData> properties = node_data->properties;
    std::stable_sort(properties.begin(),
                     properties.end(),
                     [node_data](const SceneState::PropertyData& lhs, const SceneState::PropertyData& rhs) {
                         return GetPropertyRestorePriority(node_data->type, lhs.name) <
                                GetPropertyRestorePriority(node_data->type, rhs.name);
                     });
    return properties;
}

bool IsImportedMeshSourcePath(const std::string& path) {
    const std::string extension = ToLower(GetFileExtension(path));
    return extension == "stl" ||
           extension == "dae" ||
           extension == "obj" ||
           extension == "fbx" ||
           extension == "ply" ||
           extension == "glb" ||
           extension == "gltf";
}

Ref<Material> GetImportedMeshMaterial(Node* node) {
    auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance == nullptr) {
        return {};
    }

    Ref<Mesh> mesh = mesh_instance->GetMesh();
    if (!mesh.IsValid() || mesh->IsBuiltIn() || !IsImportedMeshSourcePath(mesh->GetPath())) {
        return {};
    }

    Ref<Material> mesh_material;
    if (Ref<ArrayMesh> array_mesh = dynamic_pointer_cast<ArrayMesh>(mesh); array_mesh.IsValid()) {
        mesh_material = array_mesh->GetMaterial();
    } else if (Ref<PrimitiveMesh> primitive_mesh = dynamic_pointer_cast<PrimitiveMesh>(mesh); primitive_mesh.IsValid()) {
        mesh_material = primitive_mesh->GetMaterial();
    }

    if (mesh_material.IsValid()) {
        LOG_TRACE("PackedScene stores MeshInstance3D '{}' imported mesh material from '{}' as mesh_material.",
                  mesh_instance->GetName(),
                  mesh->GetPath());
    }

    return mesh_material;
}

bool GetAssemblyTransformOverride(Node* node, Affine3* transform) {
    auto* node_3d = Object::PointerCastTo<Node3D>(node);
    if (node_3d == nullptr) {
        return false;
    }

    auto* parent_joint = Object::PointerCastTo<Joint3D>(node->GetParent());
    return parent_joint != nullptr &&
           parent_joint->IsMotionModeEnabled() &&
           parent_joint->GetAssemblyChildTransform(node_3d, transform);
}

Variant GetTransformProperty(const Node3D* node, const Affine3& transform, const std::string& property_name) {
    if (property_name == "position") {
        return Variant(Vector3(transform.translation()));
    }
    if (property_name == "rotation_degrees") {
        const Vector3 euler = transform.GetEulerAngle(node->GetEulerOrder());
        return Variant(Vector3{
                RAD_TO_DEG(euler.x()),
                RAD_TO_DEG(euler.y()),
                RAD_TO_DEG(euler.z())});
    }
    if (property_name == "scale") {
        return Variant(transform.GetScale());
    }
    return {};
}

} // namespace

std::size_t SceneState::GetNodeCount() const {
    return nodes_.size();
}

const SceneState::NodeData* SceneState::GetNodeData(std::size_t idx) const {
    if (idx >= nodes_.size()) {
        return nullptr;
    }

    return &nodes_[idx];
}

Ref<PackedScene> SceneState::GetNodeInstance(std::size_t idx) const {
    if (idx >= nodes_.size()) {
        return {};
    }

    return nodes_[idx].instance;
}

int SceneState::AddNode(const NodeData& node) {
    nodes_.push_back(node);
    return static_cast<int>(nodes_.size() - 1);
}

void SceneState::Clear() {
    nodes_.clear();
}


PackedScene::PackedScene() {
    state_ = Ref<SceneState>(gobot::Object::New<SceneState>());
}

Ref<SceneState> PackedScene::GetState() const {
    return state_;
}

bool PackedScene::Pack(Node* root) {
    if (root == nullptr) {
        return false;
    }

    state_->Clear();

    auto pack_node = [this](Node* node, int parent, auto&& pack_node_ref) -> int {
        SceneState::NodeData node_data;
        node_data.type = node->GetClassStringName();
        node_data.name = node->GetName();
        node_data.parent = parent;
        node_data.instance = node->GetSceneInstance();

        Affine3 assembly_transform = Affine3::Identity();
        const bool has_assembly_transform = GetAssemblyTransformOverride(node, &assembly_transform);
        auto* node_3d = Object::PointerCastTo<Node3D>(node);

        auto type = Object::GetDerivedTypeByInstance(Instance(node));
        for (auto& prop : type.get_properties()) {
            if (prop.is_readonly()) {
                continue;
            }

            const std::string property_name = prop.get_name().data();
            if (property_name == "name") {
                continue;
            }

            PropertyInfo property_info;
            auto property_metadata = prop.get_metadata(PROPERTY_INFO_KEY);
            if (property_metadata.is_valid()) {
                property_info = property_metadata.get_value<PropertyInfo>();
            }

            USING_ENUM_BITWISE_OPERATORS;
            if (static_cast<bool>(property_info.usage & PropertyUsageFlags::Storage)) {
                if (has_assembly_transform &&
                    (property_name == "position" ||
                     property_name == "rotation_degrees" ||
                     property_name == "scale")) {
                    node_data.properties.push_back(
                            {property_name, GetTransformProperty(node_3d, assembly_transform, property_name)});
                } else if (property_name == "mesh_material") {
                    Ref<Material> imported_mesh_material = GetImportedMeshMaterial(node);
                    if (imported_mesh_material.IsValid()) {
                        node_data.properties.push_back({property_name, Variant(imported_mesh_material)});
                    }
                } else {
                    node_data.properties.push_back({property_name, node->Get(property_name)});
                }
            }
        }

        const int node_index = state_->AddNode(node_data);
        if (node_data.instance.IsValid()) {
            return node_index;
        }

        for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
            pack_node_ref(node->GetChild(static_cast<int>(i)), node_index, pack_node_ref);
        }

        return node_index;
    };

    pack_node(root, -1, pack_node);
    return true;
}

Node* PackedScene::Instantiate() const {
    if (!state_.IsValid() || state_->GetNodeCount() == 0) {
        LOG_ERROR("PackedScene instantiate failed: empty or invalid SceneState.");
        return nullptr;
    }

    std::vector<Node*> nodes;
    nodes.reserve(state_->GetNodeCount());
    std::vector<std::pair<Node*, SceneState::PropertyData>> deferred_properties;

    auto cleanup = [&nodes]() {
        for (Node* node : nodes) {
            if (node != nullptr && node->GetParent() == nullptr) {
                Object::Delete(node);
            }
        }
    };

    int root_index = -1;
    for (std::size_t i = 0; i < state_->GetNodeCount(); ++i) {
        const SceneState::NodeData* node_data = state_->GetNodeData(i);
        if (node_data == nullptr) {
            LOG_ERROR("PackedScene instantiate failed: missing node data at index {}.", i);
            cleanup();
            return nullptr;
        }

        Node* node = nullptr;
        if (node_data->instance.IsValid()) {
            node = node_data->instance->Instantiate();
            if (node == nullptr) {
                LOG_ERROR("PackedScene instantiate failed: cannot instantiate node '{}' from scene instance at index {}.",
                          node_data->name, i);
                cleanup();
                return nullptr;
            }
            node->SetSceneInstance(node_data->instance);
        } else {
            Type node_type = Type::get_by_name(node_data->type);
            if (!node_type.is_valid()) {
                LOG_ERROR("PackedScene instantiate failed: unknown node type '{}' at index {}.",
                          node_data->type, i);
                cleanup();
                return nullptr;
            }

            Variant new_obj = node_type.create();
            bool success = false;
            node = new_obj.convert<Node*>(&success);
            if (!success || node == nullptr) {
                LOG_ERROR("PackedScene instantiate failed: cannot create node '{}' of type '{}' at index {}.",
                          node_data->name, node_data->type, i);
                cleanup();
                return nullptr;
            }
        }

        if (!node_data->name.empty()) {
            node->SetName(node_data->name);
        }

        const std::vector<SceneState::PropertyData> properties = GetPropertiesInRestoreOrder(node_data);
        for (const auto& property : properties) {
            if (node_data->type == "Robot3D" && property.name == "mode") {
                deferred_properties.emplace_back(node, property);
                continue;
            }

            if (!node->Set(property.name, property.value)) {
                LOG_ERROR("Failed to restore property '{}' on node '{}' of type '{}'.",
                          property.name,
                          node_data->name,
                          node_data->type);
                LOG_ERROR("PackedScene instantiate aborted at node index {} while restoring property '{}'.",
                          i,
                          property.name);
                cleanup();
                Object::Delete(node);
                return nullptr;
            }
        }

        if (node_data->parent == -1) {
            if (root_index != -1) {
                LOG_ERROR("PackedScene instantiate failed: multiple root nodes, existing root index {}, duplicate index {}.",
                          root_index, i);
                cleanup();
                Object::Delete(node);
                return nullptr;
            }
            root_index = static_cast<int>(i);
        }

        nodes.push_back(node);
    }

    if (root_index == -1) {
        LOG_ERROR("PackedScene instantiate failed: no root node.");
        cleanup();
        return nullptr;
    }

    for (std::size_t i = 0; i < state_->GetNodeCount(); ++i) {
        const SceneState::NodeData* node_data = state_->GetNodeData(i);
        if (node_data->parent == -1) {
            continue;
        }

        if (node_data->parent < 0 || static_cast<std::size_t>(node_data->parent) >= nodes.size()) {
            LOG_ERROR("PackedScene instantiate failed: node index {} has invalid parent index {}.",
                      i, node_data->parent);
            cleanup();
            return nullptr;
        }

        nodes[static_cast<std::size_t>(node_data->parent)]->AddChild(nodes[i]);
    }

    for (const auto& [node, property] : deferred_properties) {
        if (!node->Set(property.name, property.value)) {
            LOG_ERROR("Failed to restore deferred property '{}' on node '{}' of type '{}'.",
                      property.name,
                      node->GetName(),
                      node->GetClassStringName());
        }
    }

    return nodes[static_cast<std::size_t>(root_index)];
}

} 

GOBOT_REGISTRATION {

    Class_<SceneState>("SceneState")
            .constructor()(CtorAsRawPtr)
            .method("get_node_count", &SceneState::GetNodeCount);

    Class_<PackedScene>("PackedScene")
            .constructor()(CtorAsRawPtr)
            .property_readonly("state", &PackedScene::GetState)
            .method("pack", &PackedScene::Pack)
            .method("instantiate", &PackedScene::Instantiate);

};
