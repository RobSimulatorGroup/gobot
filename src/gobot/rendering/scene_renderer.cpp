/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-19
*/

#include "gobot/rendering/scene_renderer.hpp"
#include "gobot/scene/resources/texture.hpp"
#include "gobot/rendering/frame_buffer_cache.hpp"
#include "gobot/editor/node3d_editor.hpp"

namespace gobot {

SceneRenderer::SceneRenderer()
{
}

SceneRenderer::~SceneRenderer() {
}

void SceneRenderer::SetRenderTarget(const RID& texture_rid) {
    view_texture_ = texture_rid;
//    auto view_frame_buffer = FrameBufferCache::GetInstance()->GetCacheFromTextures({view_texture_});
//    if (view_frame_buffer_.IsValid() && view_frame_buffer_ != view_frame_buffer) {
//        FrameBufferCache::GetInstance()->Free(view_frame_buffer_);
//    }
//    view_frame_buffer_ = view_frame_buffer;
}

void SceneRenderer::OnRenderer(const SceneTree* scene_tree) {
    DebugPass();
    FinalPass();
}

void SceneRenderer::DebugPass() {

}

void SceneRenderer::GridPass() {

}


void SceneRenderer::FinalPass() {

}



}
