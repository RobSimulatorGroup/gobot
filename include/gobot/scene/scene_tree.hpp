/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
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
