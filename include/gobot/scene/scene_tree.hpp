/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#pragma once

#include "gobot/core/object.hpp"

namespace gobot {

class Node;

class SceneTree : public Object {
    Q_OBJECT
    GOBCLASS(SceneTree, Object)
public:
    SceneTree();

    Q_SIGNALS:
    void treeChanged();

    void nodeAdded(Node *node);

    void nodeRemoved(Node *node);

    void nodeRenamed(Node *node);

};

}
