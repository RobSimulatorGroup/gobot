/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"

namespace gobot {

class Node;
class PackedScene;

class GOBOT_EXPORT SceneState : public RefCounted {
    GOBCLASS(SceneState, RefCounted)
public:
    struct PropertyData {
        std::string name;
        Variant value;
    };

    struct NodeData {
        std::string type;
        std::string name;
        int parent = -1;
        Ref<PackedScene> instance;
        std::vector<PropertyData> properties;
    };

    std::size_t GetNodeCount() const;

    const NodeData* GetNodeData(std::size_t idx) const;

    Ref<PackedScene> GetNodeInstance(std::size_t idx) const;

    int AddNode(const NodeData& node);

    void Clear();

private:
    std::vector<NodeData> nodes_;
};

class GOBOT_EXPORT PackedScene : public Resource {
    GOBCLASS(PackedScene, Resource)
public:

    PackedScene();

    Ref<SceneState> GetState() const;

    bool Pack(Node* root);

    Node* Instantiate() const;

private:
    Ref<SceneState> state_;
};

}
