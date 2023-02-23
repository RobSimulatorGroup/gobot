/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/color.hpp"
#include "gobot/rendering/render_types.hpp"

namespace gobot {


class RenderingServer : public Object {
    GOBCLASS(RenderingServer, Object)
public:
    RenderingServer();

    ~RenderingServer() override;

    static RenderingServer* GetInstance();

    // Initialize the renderer.
    void InitWindow();

    void SetViewClear(ViewId view_id,
                      ClearFlags clear_flags,
                      const Color& color = {0.f, 0.f, 0.f, 1.0},
                      float depth = 1.0f,
                      uint8_t stencil = 0);

private:
    static RenderingServer* s_singleton;

};

}
