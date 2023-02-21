/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/


#include "gobot/drivers/opengl/gl_context.hpp"

namespace gobot {

GLContext::GLContext()
{
    // glm::mat4::SetUpCoordSystem(false, false);
}

GLContext::~GLContext() = default;

void GLContext::Present()
{
}

void GLContext::OnImGui()
{
//    ImGui::TextUnformatted("%s", (const char*)(glGetString(GL_VERSION)));
//    ImGui::TextUnformatted("%s", (const char*)(glGetString(GL_VENDOR)));
//    ImGui::TextUnformatted("%s", (const char*)(glGetString(GL_RENDERER)));
}

void GLContext::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

GraphicsContext* GLContext::CreateFuncGL()
{
    return new GLContext();
}

}
