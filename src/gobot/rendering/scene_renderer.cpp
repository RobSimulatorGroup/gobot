/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-19
*/

#include "gobot/rendering/scene_renderer.hpp"
#include "gobot/scene/resources/texture.hpp"
#include "gobot/rendering/debug_draw/debug_draw.hpp"
#include "gobot/rendering/frame_buffer_cache.hpp"

namespace gobot {

SceneRenderer::SceneRenderer()
{
}

SceneRenderer::~SceneRenderer() {
}

void SceneRenderer::SetRenderTarget(const RenderRID& texture_rid) {
    view_texture_ = texture_rid;
    auto view_frame_buffer = FrameBufferCache::GetInstance()->GetCacheFromTextures({view_texture_});
    if (view_frame_buffer_.IsValid() && view_frame_buffer_ != view_frame_buffer) {
        FrameBufferCache::GetInstance()->Free(view_frame_buffer_);
        view_frame_buffer_ = view_frame_buffer;
    }
}

void SceneRenderer::OnRenderer(const SceneTree* scene_tree) {
    DebugPass();
    FinalPass();
}

void SceneRenderer::DebugPass() {

    DebugDrawEncoder dde;

    dde.Begin(0);
    dde.DrawWorldAxis(65.0);

    dde.Push();
    bx::Aabb aabb =
            {
                    {  5.0f, 1.0f, 1.0f },
                    { 10.0f, 5.0f, 5.0f },
            };
    dde.SetWireframe(true);
    dde.Draw(aabb);
    dde.Pop();

    {
        const bx::Vec3 normal = { 0.0f,  1.0f, 0.0f };
        const bx::Vec3 pos    = { 0.0f, -2.0f, 0.0f };

        bx::Plane plane(bx::init::None);
        bx::calcPlane(plane, normal, pos);

        dde.DrawGrid(Axis::Y, pos, 128, 1.0f);
    }

    dde.End();

}


void SceneRenderer::FinalPass() {
    RenderPass pass("FinalPass");
    pass.Clear();
    pass.Bind(view_frame_buffer_);
}



}
