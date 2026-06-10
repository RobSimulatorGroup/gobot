/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/rid.hpp"

#include <string>
#include <vector>

namespace gobot {

class Camera3D;
class Node;
class PhysicsWorld;
struct PhysicsSceneState;

struct DebugArrow {
    Vector3 start{Vector3::Zero()};
    Vector3 vector{Vector3::Zero()};
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
    RealType scale{1.0};
    std::string label;
};

class RendererDebugDraw {
public:
    virtual ~RendererDebugDraw() = default;

    virtual void RenderEditorDebug(const RID& render_target,
                                   const Camera3D* camera,
                                   const Node* scene_root,
                                   const PhysicsWorld* physics_world = nullptr) = 0;

    virtual void RenderDebugArrows(const RID& render_target,
                                   const Camera3D* camera,
                                   const std::vector<DebugArrow>& arrows) = 0;
};

}
