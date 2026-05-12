/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/rid.hpp"

namespace gobot {

class Camera3D;
class Node;

class RendererSceneRender {
public:
    virtual ~RendererSceneRender() = default;

    virtual void RenderScene(const RID& render_target, const Node* scene_root, const Camera3D* camera) = 0;
};

}
