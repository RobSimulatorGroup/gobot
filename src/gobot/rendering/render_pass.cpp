/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-18
*/

#include "gobot/rendering/render_pass.hpp"

#define MAX_RENDER_PASSES 1024

namespace gobot {

static ViewId& GetCounter()
{
    static ViewId id = 0;
    return id;
}

static ViewId GenerateId()
{
    auto& counter = GetCounter();
    if(counter == MAX_RENDER_PASSES - 1) {
        bgfx::frame();
        counter = 0;
    }
    ViewId idx = counter++;
    return idx;
}

RenderPass::RenderPass(const String& name)
    : view_id_(GenerateId())
{
    bgfx::resetView(view_id_);
    bgfx::setViewName(view_id_, name.toStdString().c_str());
}

void RenderPass::Bind(const FrameBuffer* fb) const {
    bgfx::setViewMode(view_id_, bgfx::ViewMode::Sequential);
    if(fb != nullptr) {
        const auto size = fb->GetSize();
        const auto width = size[0];
        const auto height = size[1];
        bgfx::setViewRect(view_id_, std::uint16_t(0), std::uint16_t(0), std::uint16_t(width), std::uint16_t(height));
        bgfx::setViewScissor(view_id_, std::uint16_t(0), std::uint16_t(0), std::uint16_t(width), std::uint16_t(height));

        bgfx::setViewFrameBuffer(view_id_, fb->GetHandle());
    } else {
        bgfx::setViewFrameBuffer(view_id_, FrameBuffer::InvalidHandle());
    }
    Touch();
}

void RenderPass::Touch() const {
    bgfx::touch(view_id_);
}

void RenderPass::Clear(RenderClearFlags clear_flags,
                       const Color& color,
                       float depth,
                       uint8_t stencil) const {
    bgfx::setViewClear(view_id_, ENUM_UINT_CAST(clear_flags), color.GetPackedRgbA(), depth, stencil);
    Touch();
}

void RenderPass::Clear() const {
    USING_ENUM_BITWISE_OPERATORS;
    Clear(RenderClearFlags::Color | RenderClearFlags::Depth | RenderClearFlags::Stencil, {0.f, 0.f, 0.f, 1.f}, 1.0f, 0);
}

void RenderPass::SetViewTransform(const Matrix4f& view, const Matrix4f& proj) const {
    bgfx::setViewTransform(view_id_, view.data(), proj.data());
}

void RenderPass::Reset() {
    auto& count = GetCounter();
    count = 0;
}

ViewId RenderPass::GetPass() {
    auto counter = GetCounter();
    if(counter == 0) {
        counter = MAX_RENDER_PASSES;
    }
    return counter - 1;
}

}
