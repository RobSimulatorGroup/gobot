#pragma once

#include "gobot/scene/node.hpp"
#include "gobot/core/io/resource.hpp"

namespace gobot {

class Node3D;
class PackedScene;

class GOBOT_EXPORT EditedScene : public Node {
    GOBCLASS(EditedScene, Node)
public:
    EditedScene();

    Node3D* GetRoot() const { return root_; }

    Ref<PackedScene> Pack() const;

    bool SaveToPath(const std::string& path) const;

    bool LoadFromPath(const std::string& path);

private:
    bool SetRoot(Node3D* root);

    Node3D* root_{nullptr};
};

}
