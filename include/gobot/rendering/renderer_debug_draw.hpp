/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/rid.hpp"

namespace gobot {

class Camera3D;
class Node;
class PhysicsWorld;
struct PhysicsSceneState;

class RendererDebugDraw {
public:
    virtual ~RendererDebugDraw() = default;

    virtual void RenderEditorDebug(const RID& render_target,
                                   const Camera3D* camera,
                                   const Node* scene_root,
                                   const PhysicsWorld* physics_world = nullptr) = 0;
};

}
