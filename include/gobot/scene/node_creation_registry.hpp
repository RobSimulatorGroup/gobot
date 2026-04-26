/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "gobot/scene/node.hpp"

namespace gobot {

struct NodeCreationEntry {
    std::string id;
    std::string display_name;
    std::string parent_id;
    std::string description;
    std::function<Node*()> create;
};

class GOBOT_EXPORT NodeCreationRegistry {
public:
    static bool RegisterNodeType(NodeCreationEntry entry);

    static const std::vector<NodeCreationEntry>& GetNodeTypes();

    static const NodeCreationEntry* FindNodeType(const std::string& id);

    static Node* CreateNode(const std::string& id);

private:
    static void EnsureBuiltInNodeTypesRegistered();
};

}
