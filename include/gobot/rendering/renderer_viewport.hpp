/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-23
 * SPDX-License-Identifier: Apache-2.0
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

    RID GetViewportRenderTarget(RID p_viewport) const;

private:
    void ViewportSetSize(Viewport *p_viewport, int p_width, int p_height, uint32_t p_view_count);

private:
    mutable RID_Owner<Viewport, true> viewport_owner_;
};


}
