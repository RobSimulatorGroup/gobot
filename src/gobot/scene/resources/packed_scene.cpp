/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#include "gobot/scene/resources/packed_scene.hpp"

#include "gobot/core/object.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/node.hpp"

namespace gobot {

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
                node_data.properties.push_back({property_name, node->Get(property_name)});
            }
        }

        const int node_index = state_->AddNode(node_data);
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
        return nullptr;
    }

    std::vector<Node*> nodes;
    nodes.reserve(state_->GetNodeCount());

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
            cleanup();
            return nullptr;
        }

        Type node_type = Type::get_by_name(node_data->type);
        if (!node_type.is_valid()) {
            cleanup();
            return nullptr;
        }

        Variant new_obj = node_type.create();
        bool success = false;
        auto* node = new_obj.convert<Node*>(&success);
        if (!success || node == nullptr) {
            cleanup();
            return nullptr;
        }

        if (!node_data->name.empty()) {
            node->SetName(node_data->name);
        }

        for (const auto& property : node_data->properties) {
            if (!node->Set(property.name, property.value)) {
                LOG_ERROR("Failed to restore property '{}' on node '{}' of type '{}'.",
                          property.name,
                          node_data->name,
                          node_data->type);
                cleanup();
                Object::Delete(node);
                return nullptr;
            }
        }

        if (node_data->parent == -1) {
            if (root_index != -1) {
                cleanup();
                Object::Delete(node);
                return nullptr;
            }
            root_index = static_cast<int>(i);
        }

        nodes.push_back(node);
    }

    if (root_index == -1) {
        cleanup();
        return nullptr;
    }

    for (std::size_t i = 0; i < state_->GetNodeCount(); ++i) {
        const SceneState::NodeData* node_data = state_->GetNodeData(i);
        if (node_data->parent == -1) {
            continue;
        }

        if (node_data->parent < 0 || static_cast<std::size_t>(node_data->parent) >= nodes.size()) {
            cleanup();
            return nullptr;
        }

        nodes[static_cast<std::size_t>(node_data->parent)]->AddChild(nodes[i]);
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
