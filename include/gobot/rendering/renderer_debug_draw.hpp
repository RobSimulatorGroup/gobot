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
struct SceneDebugData;
struct RenderViewSnapshot;

struct DebugArrow {
    Vector3 start{Vector3::Zero()};
    Vector3 vector{Vector3::Zero()};
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
    RealType scale{1.0};
    std::string label;
    bool show_head{true};
};

struct DebugLineSegment {
    Vector3 start{Vector3::Zero()};
    Vector3 end{Vector3::Zero()};
    Color color{1.0f, 1.0f, 1.0f, 1.0f};
    float width{2.0f};
};

struct DebugSphereWireframe {
    Vector3 center{Vector3::Zero()};
    RealType radius{0.05};
    Color color{0.35f, 0.85f, 1.0f, 0.65f};
    float width{1.5f};
};

struct DebugPrimitivesData {
    std::vector<DebugLineSegment> lines;
    std::vector<DebugSphereWireframe> spheres;
    bool depth_test{true};

    [[nodiscard]] bool Empty() const noexcept { return lines.empty() && spheres.empty(); }
};

class RendererDebugDraw {
public:
    virtual ~RendererDebugDraw() = default;

    virtual void RenderEditorDebug(const RID& render_target,
                                   const RenderViewSnapshot& view,
                                   const SceneDebugData& data) = 0;

    virtual void RenderDebugArrows(const RID& render_target,
                                   const RenderViewSnapshot& view,
                                   const std::vector<DebugArrow>& arrows) = 0;

    virtual void RenderDebugPrimitives(const RID& render_target,
                                       const RenderViewSnapshot& view,
                                       const DebugPrimitivesData& primitives) {
        (void)render_target;
        (void)view;
        (void)primitives;
    }
};

}
