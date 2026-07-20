/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/hash_combine.hpp"
#include "gobot/core/ref_counted.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/rid_owner.hpp"
#include "gobot/rendering/render_product.hpp"

#include <cstdint>
#include <vector>

namespace gobot {

class Image;

class RendererTextureStorage {
public:
    virtual ~RendererTextureStorage(){};

    virtual RID RenderTargetCreate() = 0;

    virtual void RenderTargetFree(RID p_rid) = 0;

    virtual void RenderTargetSetSize(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) = 0;

    virtual void RenderTargetSetOutputMask(RID p_render_target, std::uint32_t output_mask) = 0;

    virtual void* GetRenderTargetColorTextureNativeHandle(RID p_texture) = 0;

    virtual std::vector<std::uint8_t> RenderTargetReadRgbPixels(RID p_render_target, bool p_flip_y) = 0;

    virtual bool RenderTargetReadOutput(RID p_render_target,
                                        RenderOutputType output,
                                        void* destination,
                                        std::size_t destination_size,
                                        bool p_flip_y) = 0;

    // texture
    virtual RID TextureAllocate() = 0;

    virtual void Texture2DInitialize(RID texture, const Ref<Image>& image) = 0;

    virtual void TextureSetData(RID texture, const Ref<Image>& image, int layer = 0) = 0;

    virtual bool OwnsTexture(RID p_rid) const = 0;

    virtual void TextureFree(RID p_rid) = 0;
};

}
