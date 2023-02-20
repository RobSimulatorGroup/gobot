/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_render_pass.hpp"
#include "gobot/drivers/opengl/gl_frame_buffer.hpp"
#include "gobot/drivers/opengl/gl_renderer.hpp"

namespace gobot {

GLRenderPass::GLRenderPass(const RenderPassDesc& renderPassDesc)
{
    Init(renderPassDesc);
}

GLRenderPass::~GLRenderPass()
{
}

bool GLRenderPass::Init(const RenderPassDesc& renderPassDesc)
{
    m_Clear      = renderPassDesc.clear;
    m_ClearCount = renderPassDesc.attachmentCount;
    return false;
}

void GLRenderPass::BeginRenderpass(CommandBuffer* commandBuffer, float* clearColour,
                                   Framebuffer* frame, SubPassContents contents, uint32_t width, uint32_t height) const
{
    if(frame != nullptr)
    {
        frame->Bind(width, height);
        // frame->SetClearColour(clearColour);
        glClearColor(clearColour[0], clearColour[1], clearColour[2], clearColour[3]);
    }
    else
    {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glClearColor(clearColour[0], clearColour[1], clearColour[2], clearColour[3]);
        glViewport(0, 0, width, height);
    }

    if(m_Clear)
        GLRenderer::ClearInternal(RENDERER_BUFFER_COLOUR | RENDERER_BUFFER_DEPTH | RENDERER_BUFFER_STENCIL);
}

void GLRenderPass::EndRenderpass(CommandBuffer* commandBuffer)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLRenderPass::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

RenderPass* GLRenderPass::CreateFuncGL(const RenderPassDesc& renderPassDesc)
{
    return new GLRenderPass(renderPassDesc);
}

}