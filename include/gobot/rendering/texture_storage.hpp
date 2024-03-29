/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-23
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
