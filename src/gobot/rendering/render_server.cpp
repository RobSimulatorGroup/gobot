/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/rendering/texture_storage.hpp"
#include "gobot/rendering/renderer_viewport.hpp"
#include "gobot/drivers/opengl/rasterizer_gl.hpp"


namespace gobot {

RenderServer* RenderServer::s_singleton = nullptr;

RenderServer::RenderServer() {
    s_singleton =  this;
    renderer_type_ = RendererType::OpenGL46;

    RSG::viewport = new RendererViewport();
    if (renderer_type_ == RendererType::OpenGL46) {
        opengl::GLRasterizer::MakeCurrent();
    }
    RSG::rasterizer = RendererCompositor::Create();

    RSG::texture_storage = RSG::rasterizer->GetTextureStorage();
    RSG::material_storage = RSG::rasterizer->GetMaterialStorage();
}

RendererType RenderServer::GetRendererType() {
    return renderer_type_;
}

RenderServer::~RenderServer() {
    s_singleton = nullptr;
    delete RSG::rasterizer;
    delete RSG::viewport;
}

RenderServer* RenderServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize RenderingServer");
    return s_singleton;
}

void RenderServer::Draw() {
    SceneTree::GetInstance()->GetRoot()->GetWindow()->SwapBuffers();
}

void RenderServer::Free(const RID& p_rid) {
    if (p_rid.IsNull()) [[unlikely]] {
        return;
    }
    if (RSG::viewport->Free(p_rid)) {
        return;
    }
    if (RSG::utilities->Free(p_rid)) {
        return;
    }
}

void* RenderServer::GetRenderTargetColorTextureNativeHandle(const RID& p_view_port) {
    return RSG::viewport->GetRenderTargetColorTextureNativeHandle(p_view_port);
}


RID RenderServer::CreateMesh() {
    return {};
}



}
