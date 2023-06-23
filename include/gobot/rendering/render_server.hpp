/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/rid.hpp"
#include "render_types.hpp"
#include "rendering_server_globals.hpp"
#include "renderer_viewport.hpp"

namespace gobot {

#define RS RenderServer


class GOBOT_EXPORT RenderServer : public Object {
    GOBCLASS(RenderServer, Object)
public:
    RenderServer();

    ~RenderServer() override;

    RendererType GetRendererType();

    static RenderServer* GetInstance();

    // viewport
    RID ViewportCreate() {
        RID viewport = RSG::viewport->ViewportAllocate();
        RSG::viewport->ViewportInitialize(viewport);
        return viewport;
    }

    void ViewportSetSize(const RID& p_rid, int width, int height) {
        RSG::viewport->ViewportSetSize(p_rid, width, height);
    }

    void* GetRenderTargetColorTextureNativeHandle(const RID& p_render_target);

    void Free(const RID& rid);

    RID CreateMesh();

    void Draw();

private:
    static RenderServer* s_singleton;

    RendererType renderer_type_;

};

}
