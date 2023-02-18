/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
 * This file is modified by Zikun Yu, 23-1-20
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/core/os/main_loop.hpp"

namespace gobot {

class Node;

class GOBOT_EXPORT SceneTree : public MainLoop {
    Q_OBJECT
    GOBCLASS(SceneTree, MainLoop)

public:
    SceneTree();

    ~SceneTree() override;

    void Initialize() override;

    void Finalize() override;

    [[nodiscard]] FORCE_INLINE Node* GetRoot() const { return root; }

    [[nodiscard]] int GetNodeCount() const;

    static SceneTree *GetInstance() {
        return singleton;
    }

Q_SIGNALS:
    void treeChanged();
    void nodeAdded(Node *node);
    void nodeRemoved(Node *node);
    void nodeRenamed(Node *node);

private:
    Node *root = nullptr;
    int node_count = 0;

    static SceneTree *singleton;

    friend class Node;
};

}
