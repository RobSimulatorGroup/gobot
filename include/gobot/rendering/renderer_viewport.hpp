/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-23
*/

#pragma once

#include "gobot/core/rid_owner.hpp"
#include "gobot/core/math/matrix.hpp"

namespace gobot {

class RendererViewport {

    struct Viewport {
        RID self;
        RID render_target;

        Vector2i size;
        uint32_t view_count;
        bool viewport_render_direct_to_screen;
    };

public:
    RendererViewport();

    virtual ~RendererViewport() {}

    RID ViewportAllocate();

    void ViewportInitialize(RID p_rid);

    void ViewportSetSize(RID p_viewport, int p_width, int p_height);

    bool Free(const RID& p_rid);

    void* GetRenderTargetColorTextureNativeHandle(RID p_viewport);

private:
    void ViewportSetSize(Viewport *p_viewport, int p_width, int p_height, uint32_t p_view_count);

private:
    mutable RID_Owner<Viewport, true> viewport_owner_;
};


}
