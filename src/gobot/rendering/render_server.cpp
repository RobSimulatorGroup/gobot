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

#include <bgfx/bgfx.h>

namespace gobot {

RenderServer* RenderServer::s_singleton = nullptr;

RenderServer::RenderServer() {
    s_singleton =  this;
}

RenderServer::~RenderServer() {
    s_singleton = nullptr;
}

RenderServer* RenderServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize RenderingServer");
    return s_singleton;
}

void RenderServer::InitWindow() {
    auto window = SceneTree::GetInstance()->GetRoot()->GetWindowsInterface();
    RenderInitProps init;
    init.type     = bgfx::RendererType::Count; // auto select
    init.vendorId = ENUM_UINT_CAST(VendorID::None); // auto select
    init.platformData.nwh  = window->GetNativeWindowHandle();
    init.platformData.ndt  = window->GetNativeDisplayHandle();
    init.resolution.width  = window->GetHeight();
    init.resolution.height = window->GetWidth();
    init.resolution.reset  = ENUM_UINT_CAST(RenderResetFlags::Vsync); //  Enable V-Sync.

    bgfx::init(init);
};

void RenderServer::SetViewClear(ViewId view_id, ClearFlags clear_flags, const Color& color, float depth, uint8_t stencil) {
    bgfx::setViewClear(view_id, std::underlying_type_t<ClearFlags>(clear_flags), color.GetPackedRgbA());
};

void RenderServer::SetViewRect(ViewId id, uint16_t x, uint16_t y, uint16_t width, uint16_t height) {
    bgfx::setViewRect(id, x, y, width, height);
}

void RenderServer::SetDebug(RenderDebugFlags debug_flags) {
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
    bgfx::reset(width, height, ENUM_UINT_CAST(reset_flags), bgfx::TextureFormat::Enum(format));
}

void RenderServer::Touch(ViewId id) {
    bgfx::touch(id);
}


const RenderStats* RenderServer::GetStats() {
    return bgfx::getStats();
}


}
