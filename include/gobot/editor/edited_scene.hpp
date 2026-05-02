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

    bool NewScene();

    Ref<PackedScene> Pack() const;

    bool SaveToPath(const std::string& path) const;

    bool LoadFromPath(const std::string& path);

    bool LoadFromPath(const std::string& path, bool default_robot_motion_mode);

    Node3D* AddSceneFromPath(const std::string& path);

private:
    bool SetRoot(Node3D* root);

    Node3D* root_{nullptr};
};

}
