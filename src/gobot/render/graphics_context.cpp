/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#include "gobot/render/graphics_context.hpp"
#include "gobot/error_macros.hpp"

#include "gobot/drivers/opengl/gl_functions.hpp"

namespace gobot {

GraphicsContext* (*GraphicsContext::CreateFunc)() = nullptr;

RenderAPI GraphicsContext::s_RenderAPI;

GraphicsContext* GraphicsContext::Create()
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No GraphicsContext Create Function");
    return CreateFunc();
}

void GraphicsContext::SetRenderAPI(RenderAPI api)
{
    s_RenderAPI = api;

    switch(s_RenderAPI)
    {
        case RenderAPI::OpenGL:
            OpenGLMakeDefault();
            break;
        default:
            break;
    }
}

}
