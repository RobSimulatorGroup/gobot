/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_pipeline.hpp"
#include "gobot/drivers/opengl/gl_render_pass.hpp"
#include "gobot/drivers/opengl/gl_command_buffer.hpp"
#include "gobot/drivers/opengl/gl_shader.hpp"
#include "gobot/drivers/opengl/gl_texture.hpp"
#include "gobot/drivers/opengl/gl_renderer.hpp"
#include "gobot/drivers/opengl/gl_frame_buffer.hpp"
#include "gobot/drivers/opengl/gl.hpp"

namespace gobot {

GLPipeline::GLPipeline(const PipelineDesc& pipelineDesc)
        : m_RenderPass(nullptr)
{
    Init(pipelineDesc);
}

GLPipeline::~GLPipeline()
{
    glDeleteVertexArrays(1, &m_VertexArray);
}

void VertexAtrribPointer(RHIFormat format, uint32_t index, size_t offset, uint32_t stride)
{
    switch(format)
    {
        case RHIFormat::R32_Float:
            glVertexAttribPointer(index, 1, GL_FLOAT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32_Float:
            glVertexAttribPointer(index, 2, GL_FLOAT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32B32_Float:
            glVertexAttribPointer(index, 3, GL_FLOAT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32B32A32_Float:
            glVertexAttribPointer(index, 4, GL_FLOAT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R8_UInt:
            glVertexAttribPointer(index, 1, GL_UNSIGNED_BYTE, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32_UInt:
            glVertexAttribPointer(index, 1, GL_UNSIGNED_INT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32_UInt:
            glVertexAttribPointer(index, 2, GL_UNSIGNED_INT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32B32_UInt:
            glVertexAttribPointer(index, 3, GL_UNSIGNED_INT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32B32A32_UInt:
            glVertexAttribPointer(index, 4, GL_UNSIGNED_INT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32_Int:
            glVertexAttribPointer(index, 2, GL_INT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32B32_Int:
            glVertexAttribPointer(index, 3, GL_INT, false, stride, (const void*)(intptr_t)(offset));
            break;
        case RHIFormat::R32G32B32A32_Int:
            glVertexAttribPointer(index, 4, GL_INT, false, stride, (const void*)(intptr_t)(offset));
            break;
    }
}

bool GLPipeline::Init(const PipelineDesc& pipelineDesc)
{
    m_TransparencyEnabled = pipelineDesc.transparencyEnabled;
    m_CullMode            = pipelineDesc.cullMode;
    m_Description         = pipelineDesc;

    glGenVertexArrays(1, &m_VertexArray);

    m_Shader    = pipelineDesc.shader.get();
    m_BlendMode = pipelineDesc.blendMode;

    CreateFramebuffers();
    return true;
}

void GLPipeline::BindVertexArray()
{
    glBindVertexArray(m_VertexArray);
    auto& vertexLayout = ((GLShader*)m_Shader)->GetBufferLayout().GetLayout();
    uint32_t count     = 0;

    for(auto& layout : vertexLayout)
    {
        glEnableVertexAttribArray(count);
        size_t offset = static_cast<size_t>(layout.offset);
        VertexAtrribPointer(layout.format, count, offset, ((GLShader*)m_Shader)->GetBufferLayout().GetStride());
        count++;
    }
}

void GLPipeline::CreateFramebuffers()
{
    std::vector<TextureType> attachmentTypes;
    std::vector<Texture*> attachments;

    if(m_Description.swapchainTarget)
    {
        attachmentTypes.push_back(TextureType::COLOUR);
//        attachments.push_back(Renderer::GetMainSwapChain()->GetImage(0));
    }
    else
    {
        for(auto texture : m_Description.colourTargets)
        {
            if(texture)
            {
                attachmentTypes.push_back(texture->GetType());
                attachments.push_back(texture);
            }
        }
    }

    if(m_Description.depthTarget)
    {
        attachmentTypes.push_back(m_Description.depthTarget->GetType());
        attachments.push_back(m_Description.depthTarget);
    }

    if(m_Description.depthArrayTarget)
    {
        attachmentTypes.push_back(m_Description.depthArrayTarget->GetType());
        attachments.push_back(m_Description.depthArrayTarget);
    }

    RenderPassDesc renderPassDesc;
    renderPassDesc.attachmentCount = uint32_t(attachmentTypes.size());
    renderPassDesc.attachmentTypes = attachmentTypes.data();
    renderPassDesc.attachments     = attachments.data();
    renderPassDesc.clear           = m_Description.clearTargets;
    renderPassDesc.cubeMapIndex    = m_Description.cubeMapIndex;
    renderPassDesc.mipIndex        = m_Description.mipIndex;

    m_RenderPass = RenderPass::Get(renderPassDesc);

    FramebufferDesc frameBufferDesc {};
    frameBufferDesc.width           = GetWidth();
    frameBufferDesc.height          = GetHeight();
    frameBufferDesc.attachmentCount = uint32_t(attachments.size());
    frameBufferDesc.renderPass      = m_RenderPass.Get();
    frameBufferDesc.attachmentTypes = attachmentTypes.data();

    if(m_Description.swapchainTarget)
    {
//        for(uint32_t i = 0; i < Renderer::GetMainSwapChain()->GetSwapChainBufferCount(); i++)
//        {
//            frameBufferDesc.screenFBO   = true;
//            attachments[0]              = Renderer::GetMainSwapChain()->GetImage(i);
//            frameBufferDesc.attachments = attachments.data();
//
//            m_Framebuffers.emplace_back(Framebuffer::Get(frameBufferDesc));
//        }
    }
    else if(m_Description.depthArrayTarget)
    {
        for(uint32_t i = 0; i < ((GLTextureDepthArray*)m_Description.depthArrayTarget)->GetCount(); ++i)
        {
            frameBufferDesc.layer     = i;
            frameBufferDesc.screenFBO = false;

            attachments[0]              = m_Description.depthArrayTarget;
            frameBufferDesc.attachments = attachments.data();

            m_Framebuffers.emplace_back(Framebuffer::Get(frameBufferDesc));
        }
    }
    else
    {
        frameBufferDesc.attachments = attachments.data();
        frameBufferDesc.screenFBO   = false;
        m_Framebuffers.emplace_back(Framebuffer::Get(frameBufferDesc));
    }
}

void GLPipeline::Bind(CommandBuffer* commandBuffer, uint32_t layer)
{
//    GLRenderer::Instance()->GetBoundPipeline() = this;

    Framebuffer* framebuffer;

    if(m_Description.swapchainTarget)
    {
//        framebuffer = m_Framebuffers[Renderer::GetMainSwapChain()->GetCurrentBufferIndex()];
    }
    else if(m_Description.depthArrayTarget)
    {
        framebuffer = m_Framebuffers[layer].Get();
    }
    else
    {
        framebuffer = m_Framebuffers[0].Get();
    }

    m_RenderPass->BeginRenderpass(commandBuffer, m_Description.clearColour, framebuffer, SubPassContents::INLINE, GetWidth(), GetHeight());

    m_Shader->Bind();

    if(m_TransparencyEnabled)
    {
        glEnable(GL_BLEND);

        glBlendEquation(GL_FUNC_ADD);

        if(m_BlendMode == BlendMode::SrcAlphaOneMinusSrcAlpha)
        {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        else if(m_BlendMode == BlendMode::ZeroSrcColor)
        {
            glBlendFunc(GL_ZERO, GL_SRC_COLOR);
        }
        else if(m_BlendMode == BlendMode::OneZero)
        {
            glBlendFunc(GL_ONE, GL_ZERO);
        }
        else
        {
            glBlendFunc(GL_NONE, GL_NONE);
        }
    }
    else
        glDisable(GL_BLEND);

    glEnable(GL_CULL_FACE);

    switch(m_CullMode)
    {
        case CullMode::BACK:
            glCullFace(GL_BACK);
            break;
        case CullMode::FRONT:
            glCullFace(GL_FRONT);
            break;
        case CullMode::FRONTANDBACK:
            glCullFace(GL_FRONT_AND_BACK);
            break;
        case CullMode::NONE:
            glDisable(GL_CULL_FACE);
            break;
    }

    glFrontFace(GL_CCW);

    if(m_LineWidth != 1.0f)
        glLineWidth(m_LineWidth);
}

void GLPipeline::End(CommandBuffer* commandBuffer)
{
    m_RenderPass->EndRenderpass(commandBuffer);

    if(m_LineWidth != 1.0f)
        glLineWidth(1.0f);

    GLRenderer::Instance()->GetBoundPipeline() = nullptr;
}

void GLPipeline::ClearRenderTargets(CommandBuffer* commandBuffer)
{
    for(auto framebuffer : m_Framebuffers)
    {
        framebuffer->Bind();
        GLRenderer::ClearInternal(RENDERER_BUFFER_COLOUR | RENDERER_BUFFER_DEPTH | RENDERER_BUFFER_STENCIL);
    }
}

void GLPipeline::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

Pipeline* GLPipeline::CreateFuncGL(const PipelineDesc& pipelineDesc)
{
    return new GLPipeline(pipelineDesc);
}

}
