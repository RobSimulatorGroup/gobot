/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-26
*/

#include "gobot/rendering/debug_draw_renderer.hpp"
#include "gobot/rendering/render_types.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"

namespace gobot {


DebugDrawRenderer* DebugDrawRenderer::s_singleton = nullptr;

DebugDrawRenderer::DebugDrawRenderer() {
    s_singleton = this;
}

DebugDrawRenderer::~DebugDrawRenderer() {
    s_singleton = nullptr;
}

DebugDrawRenderer* DebugDrawRenderer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize DebugDrawRenderer");
    return s_singleton;
}

}
