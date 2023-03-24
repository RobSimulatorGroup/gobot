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

#include <bgfx/bgfx.h>

namespace gobot {

RenderServer* RenderServer::s_singleton = nullptr;

RenderServer::RenderServer() {
    s_singleton =  this;
    debug_flags_ = RenderDebugFlags::None;
    reset_flags_ = RenderResetFlags::Vsync;

    RSG::compositor = new RendererCompositor();

    RSG::texture_storage = RSG::compositor->GetTextureStorage();
}

bool RenderServer::HasInit() {
    return s_singleton != nullptr;
}

RendererType RenderServer::GetRendererType() {
    return RendererType(bgfx::getRendererType());
}

RenderServer::~RenderServer() {
    s_singleton = nullptr;

    delete RSG::compositor;
}

RenderServer* RenderServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize RenderingServer");
    return s_singleton;
}

void RenderServer::ShutDown() {
    bgfx::shutdown();
}

void RenderServer::InitWindow() {
    auto window = SceneTree::GetInstance()->GetRoot()->GetWindow();
    RenderInitProps init;
    init.type     = bgfx::RendererType::Count; // auto select
    init.vendorId = ENUM_UINT_CAST(VendorID::None); // auto select
    init.platformData.nwh  = window->GetNativeWindowHandle();
    init.platformData.ndt  = window->GetNativeDisplayHandle();
    init.resolution.width  = window->GetWidth();
    init.resolution.height = window->GetHeight();
    init.resolution.reset  = ENUM_UINT_CAST(reset_flags_); //  Enable V-Sync.

    bgfx::init(init);
};


void RenderServer::SetDebug(RenderDebugFlags debug_flags) {
    debug_flags_ = debug_flags;
    bgfx::setDebug(ENUM_UINT_CAST(debug_flags));
}

void RenderServer::DebugTextClear() {
    bgfx::dbgTextClear();
}

void RenderServer::DebugTextImage(uint16_t x, uint16_t y, uint16_t width, uint16_t height, const void* data, uint16_t pitch) {
    bgfx::dbgTextImage(x, y, width, height, data, pitch);
}

void RenderServer::DebugTextPrintf(uint16_t x, uint16_t y, uint8_t attr, const char* format, ...) {
    va_list argList;
    va_start(argList, format);
    bgfx::dbgTextPrintfVargs(x, y, attr, format, argList);
    va_end(argList);
}

uint32_t RenderServer::Frame(bool capture) {
    return bgfx::frame(capture);
}

void RenderServer::Reset(uint32_t width, uint32_t height, RenderResetFlags reset_flags, TextureFormat format) {
    reset_flags_ = reset_flags;
    bgfx::reset(width, height, ENUM_UINT_CAST(reset_flags), bgfx::TextureFormat::Enum(format));
}

void RenderServer::Touch(ViewId id) {
    bgfx::touch(id);
}

const RenderStats* RenderServer::GetStats() {
    return bgfx::getStats();
}

RenderRID RenderServer::CreateTexture2D(uint16_t width,
                                        uint16_t height,
                                        bool has_mips,
                                        uint16_t num_layers,
                                        TextureFormat format,
                                        TextureFlags flags,
                                        const MemoryView* mem) {
    return RSG::texture_storage->CreateTexture2D(width, height, has_mips, num_layers, format, flags, mem);
}

RenderRID RenderServer::CreateTexture3D(uint16_t width,
                                        uint16_t height,
                                        uint16_t depth,
                                        bool has_mips,
                                        TextureFormat format,
                                        TextureFlags flags,
                                        const MemoryView* mem) {
    return RSG::texture_storage->CreateTexture3D(width, height, depth, has_mips, format, flags, mem);
}

RenderRID RenderServer::CreateTextureCube(uint16_t size,
                                          bool has_mips,
                                          uint16_t num_layers,
                                          TextureFormat format,
                                          TextureFlags flags,
                                          const MemoryView* mem) {
    bgfx::TextureHandle handle = bgfx::createTextureCube(size, has_mips, num_layers,
                                                       bgfx::TextureFormat::Enum(format),
                                                       ENUM_UINT_CAST(flags), mem);
    return RenderRID::FromUint16(handle.idx);
}

bool RenderServer::FreeTexture(const RenderRID& rid) {
    return RSG::texture_storage->Free(rid);
}


}
