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

RenderingServer* RenderingServer::s_singleton = nullptr;

RenderingServer::RenderingServer() {
    s_singleton =  this;
}

RenderingServer::~RenderingServer() {
    s_singleton = nullptr;
}

RenderingServer* RenderingServer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize RenderingServer");
    return s_singleton;
}

void RenderingServer::InitWindow() {
    auto window = SceneTree::GetInstance()->GetRoot()->GetWindowsInterface();
    bgfx::Init init;
    init.type     = bgfx::RendererType::Count; // auto select
    init.vendorId = BGFX_PCI_ID_NONE; // auto select
    init.platformData.nwh  = window->GetNativeWindowHandle();
    init.platformData.ndt  = window->GetNativeDisplayHandle();
    init.resolution.width  = window->GetHeight();
    init.resolution.height = window->GetWidth();
    init.resolution.reset  = BGFX_RESET_VSYNC; //  Enable V-Sync.

    bgfx::init(init);

    bgfx::setDebug(BGFX_DEBUG_TEXT);

    USING_ENUM_BITWISE_OPERATORS;
    SetViewClear(0, ClearFlags::Color|ClearFlags::Depth);

};

void RenderingServer::SetViewClear(ViewId view_id, ClearFlags clear_flags, const Color& color, float depth, uint8_t stencil) {
    bgfx::setViewClear(view_id, std::underlying_type_t<ClearFlags>(clear_flags), color.PackedRgbA());
};


}
