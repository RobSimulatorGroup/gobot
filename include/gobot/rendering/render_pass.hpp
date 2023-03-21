/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-18
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/rendering/render_types.hpp"
#include "gobot/rendering/frame_buffer.hpp"
#include "gobot/core/color.hpp"

namespace gobot {

/*
 * Views are the primary sorting mechanism in bgfx. They represent buckets of draw and compute calls, or what are often known as ‘passes’.
   When compute calls and draw calls occupy the same bucket, the compute calls will be sorted to execute first.
   Compute calls are always executed in order of submission, while draw calls are sorted by internal state if the View is not in sequential mode.
   In most cases where the z-buffer is used, this change in order does not affect the desired output.
   When draw call order needs to be preserved (e.g. when rendering GUIs), Views can be set to use sequential mode with bgfx::setViewMode.
   Sequential order is less efficient, because it doesn’t allow state change optimization, and should be avoided when possible.

   By default, Views are sorted by their View ID, in ascending order.
   For dynamic renderers where the right order might not be known until the last moment,
   View IDs can be changed to use arbitrary ordering with bgfx::setViewOrder.
 */

class RenderPass
{
public:
    explicit RenderPass(const String& name);

    void Bind(const FrameBuffer* fb = nullptr) const;

    void Touch() const;

    void Clear(RenderClearFlags clear_flags,
               const Color& color = {0.f, 0.f, 0.f, 1.0},
               float depth = 1.0f,
               uint8_t stencil = 0) const;

    void Clear() const;

    void SetViewTransform(const Matrix4f& view, const Matrix4f& proj) const;

    static void Reset();

    static ViewId GetPass();

private:
    ViewId view_id_;
};

}
