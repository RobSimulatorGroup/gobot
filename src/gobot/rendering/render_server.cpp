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


namespace gobot {

RenderServer* RenderServer::s_singleton = nullptr;

RenderServer::RenderServer() {
    s_singleton =  this;

    RSG::compositor = new RendererCompositor();

    RSG::texture_storage = RSG::compositor->GetTextureStorage();
}

bool RenderServer::HasInit() {
    return s_singleton != nullptr;
}

RendererType RenderServer::GetRendererType() {
    return RendererType();
}

RenderServer::~RenderServer() {
    s_singleton = nullptr;

    delete RSG::compositor;
}

RenderServer* RenderServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize RenderingServer");
    return s_singleton;
}

void RenderServer::InitWindow() {
    auto window = SceneTree::GetInstance()->GetRoot()->GetWindow();
//    RenderInitProps init;
//    init.type     = bgfx::RendererType::Count; // auto select
//    init.vendorId = ENUM_UINT_CAST(VendorID::None); // auto select
//    init.platformData.nwh  = window->GetNativeWindowHandle();
//    init.platformData.ndt  = window->GetNativeDisplayHandle();
//    init.resolution.width  = window->GetWidth();
//    init.resolution.height = window->GetHeight();
//    init.resolution.reset  = ENUM_UINT_CAST(reset_flags_); //  Enable V-Sync.
};


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
    return RSG::texture_storage->Free(rid);
}

RID RenderServer::CreateMesh() {
    return {};
}

bool RenderServer::FreeMesh(const RID& rid) {
    return false;
}


}
