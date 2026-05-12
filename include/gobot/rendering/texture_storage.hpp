/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/hash_combine.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/rid_owner.hpp"

namespace gobot {

class RendererTextureStorage {
public:
    virtual ~RendererTextureStorage(){};

    virtual RID RenderTargetCreate() = 0;

    virtual void RenderTargetFree(RID p_rid) = 0;

    virtual void RenderTargetSetSize(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) = 0;

    virtual void* GetRenderTargetColorTextureNativeHandle(RID p_texture) = 0;

    // texture
    virtual RID TextureAllocate() = 0;

    virtual void TextureFree(RID p_rid) = 0;
};

}
