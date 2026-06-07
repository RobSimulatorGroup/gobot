#pragma once

#include "gobot_export.h"

namespace gobot {

class Node;
class SceneTree;

class GOBOT_EXPORT RuntimeSceneOwner {
public:
    RuntimeSceneOwner() = default;
    ~RuntimeSceneOwner();

    RuntimeSceneOwner(const RuntimeSceneOwner&) = delete;
    RuntimeSceneOwner& operator=(const RuntimeSceneOwner&) = delete;

    void Attach(Node* scene_root);
    void Clear();

private:
    SceneTree* runtime_tree_{nullptr};
    Node* scene_root_{nullptr};
};

} // namespace gobot
