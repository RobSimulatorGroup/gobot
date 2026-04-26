#pragma once

#include "gobot/scene/node.hpp"

namespace gobot {

class Node3D;

class GOBOT_EXPORT EditedScene : public Node {
    GOBCLASS(EditedScene, Node)
public:
    EditedScene();

    Node3D* GetRoot() const { return root_; }

private:
    Node3D* root_{nullptr};
};

}
