/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-12
*/

#include "gobot/drivers/opengl/gl_imgui_renderer.hpp"
#include "gobot/drivers/opengl/gl.hpp"

#include "imgui.h"
#include "imgui_impl_opengl3.h"

namespace gobot {

GLIMGUIRenderer::GLIMGUIRenderer(std::uint32_t width, std::uint32_t height, bool clear_screen)
    : window_handle_(nullptr),
      clear_screen_(clear_screen)
{
    ImGui_ImplOpenGL3_Init("#version 410");
    ImGui_ImplOpenGL3_NewFrame();
}

GLIMGUIRenderer::~GLIMGUIRenderer()
{
    ImGui_ImplOpenGL3_Shutdown();
}

void GLIMGUIRenderer::Init()
{
}

void GLIMGUIRenderer::NewFrame()
{
}

void GLIMGUIRenderer::Render(CommandBuffer* commandBuffer)
{
    ImGui::Render();

    if(clear_screen_)
    {
//        GLCall(glClear(GL_COLOR_BUFFER_BIT));
    }
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLIMGUIRenderer::OnResize(uint32_t width, uint32_t height)
{
}

void GLIMGUIRenderer::MakeDefault()
{
//    CreateFunc = CreateFuncGL;
}

IMGUIRenderer* GLIMGUIRenderer::CreateFuncGL(uint32_t width, uint32_t height, bool clearScreen)
{
    return new GLIMGUIRenderer(width, height, clearScreen);
}

void GLIMGUIRenderer::RebuildFontTexture()
{
    ImGui_ImplOpenGL3_DestroyFontsTexture();
    ImGui_ImplOpenGL3_CreateFontsTexture();
}

}
