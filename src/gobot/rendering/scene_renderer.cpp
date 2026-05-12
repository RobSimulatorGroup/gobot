/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-19
 * SPDX-License-Identifier: Apache-2.0
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
