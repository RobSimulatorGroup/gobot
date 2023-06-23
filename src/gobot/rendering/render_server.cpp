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
#include "gobot/rendering/scene_viewport.hpp"
#include "gobot/drivers/opengl/rasterizer_gles3.hpp"


namespace gobot {

RenderServer* RenderServer::s_singleton = nullptr;

RenderServer::RenderServer() {
    s_singleton =  this;
    renderer_type_ = RendererType::OpenGL46;

    RSG::viewport = new RendererViewport();
}

bool RenderServer::HasInit() {
    return s_singleton != nullptr;
}

RendererType RenderServer::GetRendererType() {
    return renderer_type_;
}

RenderServer::~RenderServer() {
    s_singleton = nullptr;

    delete RSG::compositor;
    delete RSG::viewport;
}

RenderServer* RenderServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize RenderingServer");
    return s_singleton;
}

void RenderServer::InitWindow() {
    if (renderer_type_ == RendererType::OpenGL46) {
        opengl::RasterizerGLES3::MakeCurrent();
    }
};

void RenderServer::Draw() {
    SceneTree::GetInstance()->GetRoot()->GetWindow()->SwapBuffers();
}


//RID RenderServer::CreateTexture2D(uint16_t width,
//                                  uint16_t height,
//                                  bool has_mips,
//                                  uint16_t num_layers,
//                                  TextureFormat format,
//                                  TextureFlags flags) {
//    return RSG::texture_storage->CreateTexture2D(width, height, has_mips, num_layers, format, flags);
//}
//
//RID RenderServer::CreateTexture3D(uint16_t width,
//                                  uint16_t height,
//                                  uint16_t depth,
//                                  bool has_mips,
//                                  TextureFormat format,
//                                  TextureFlags flags) {
//    return RSG::texture_storage->CreateTexture3D(width, height, depth, has_mips, format, flags);
//}
//
//RID RenderServer::CreateTextureCube(uint16_t size,
//                                    bool has_mips,
//                                    uint16_t num_layers,
//                                    TextureFormat format,
//                                    TextureFlags flags) {
//    return RSG::texture_storage->CreateTextureCube(width, height, depth, has_mips, format, flags);
//}

bool RenderServer::FreeTexture(const RID& rid) {
//    return RSG::texture_storage->Free(rid);
}

RID RenderServer::CreateMesh() {
    return {};
}

bool RenderServer::FreeMesh(const RID& rid) {
    return false;
}


}
