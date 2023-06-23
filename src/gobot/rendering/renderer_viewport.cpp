/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-23
*/

#include "gobot/rendering/renderer_viewport.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"

namespace gobot {

RendererViewport::RendererViewport() {

}

RID RendererViewport::ViewportAllocate() {
    return viewport_owner_.AllocateRID();
}

void RendererViewport::ViewportInitialize(RID p_rid) {
    viewport_owner_.InitializeRID(p_rid);
    Viewport *viewport = viewport_owner_.GetOrNull(p_rid);
    viewport->self = p_rid;
    viewport->render_target = RSG::texture_storage->RenderTargetCreate();
    viewport->viewport_render_direct_to_screen = false;
}

void RendererViewport::ViewportSetSize(RID p_viewport, int p_width, int p_height) {
    ERR_FAIL_COND(p_width < 0 || p_height < 0);

    Viewport *viewport = viewport_owner_.GetOrNull(p_viewport);
    ERR_FAIL_COND(!viewport);

    ViewportSetSize(viewport, p_width, p_height, 1);
}

void RendererViewport::ViewportSetSize(Viewport *p_viewport, int p_width, int p_height, uint32_t p_view_count) {
    Vector2i new_size(p_width, p_height);
    if (p_viewport->size != new_size || p_viewport->view_count != p_view_count) {
        p_viewport->size = new_size;
        p_viewport->view_count = p_view_count;

        RSG::texture_storage->RenderTargetSetSize(p_viewport->render_target, p_width, p_height, p_view_count);
    }
}

bool RendererViewport::Free(const RID& p_rid) {
    if (viewport_owner_.Owns(p_rid)) {
        Viewport *viewport = viewport_owner_.GetOrNull(p_rid);

        RSG::texture_storage->RenderTargetFree(viewport->render_target);

        viewport_owner_.Free(p_rid);

        return true;
    }

    return false;
}

void* RendererViewport::GetRenderTargetColorTextureNativeHandle(RID p_viewport) {
    Viewport *viewport = viewport_owner_.GetOrNull(p_viewport);

    return RSG::texture_storage->GetRenderTargetColorTextureNativeHandle(viewport->render_target);
}


}
