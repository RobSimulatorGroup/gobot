/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
 * This file is modified by Zikun Yu, 23-1-20
*/


#include "gobot/scene/scene_tree.hpp"

namespace gobot {

SceneTree *SceneTree::singleton = nullptr;

void SceneTree::Initialize()
{

}

void SceneTree::Finalize()
{

}

int SceneTree::GetNodeCount() const {
    return node_count;
}

SceneTree::SceneTree() {
    if (singleton == nullptr) {
        singleton = this;
    }

    root = Node::New<Node>();
    root->SetName("root");
    root->SetTree(this);
}

SceneTree::~SceneTree() {
    if (root) {
        root->SetTree(nullptr);
        root->PropagateAfterExitTree();
        Node::Delete(root);
    }

    if (singleton == this) {
        singleton = nullptr;
    }
}

} // End of namespace gobot
