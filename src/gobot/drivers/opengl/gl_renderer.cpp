/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_renderer.hpp"
#include "gobot/drivers/opengl/gl_vertex_buffer.hpp"
#include "gobot/drivers/opengl/gl_index_buffer.hpp"
#include "gobot/drivers/opengl/gl_utilities.hpp"
#include "gobot/drivers/opengl/gl_descriptor_set.hpp"
#include "gobot/drivers/opengl/gl_texture.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/drivers/opengl/gl.hpp"
#include "gobot/render/frame_buffer.hpp"


namespace gobot {

GLRenderer::GLRenderer()
{
    m_RendererTitle = "OPENGL";
    auto& caps      = Renderer::GetCapabilities();

    caps.Vendor   = (const char*)glGetString(GL_VENDOR);
    caps.Renderer = (const char*)glGetString(GL_RENDERER);
    caps.Version  = (const char*)glGetString(GL_VERSION);

    glGetIntegerv(GL_MAX_SAMPLES, &caps.MaxSamples);
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &caps.MaxAnisotropy);
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &caps.MaxTextureUnits);
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &caps.UniformBufferOffsetAlignment);

    m_DefaultVertexBuffer = new GLVertexBuffer(BufferUsage::STATIC);
    uint16_t data[1];
    m_DefaultIndexBuffer = new GLIndexBuffer(data, 0, BufferUsage::STATIC);
}

GLRenderer::~GLRenderer()
{
    delete m_DefaultVertexBuffer;
    delete m_DefaultIndexBuffer;
}

void GLRenderer::InitInternal()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
}

void GLRenderer::Begin()
{
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void GLRenderer::ClearInternal(uint32_t buffer)
{
    glClear(GLUtilities::RendererBufferToGL(buffer));
}

void GLRenderer::PresentInternal()
{
}

void GLRenderer::PresentInternal(CommandBuffer* commandBuffer)
{
}

void GLRenderer::SetDepthTestingInternal(bool enabled)
{
    if(enabled) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
}

void GLRenderer::SetDepthMaskInternal(bool enabled)
{
    glDepthMask(enabled ? GL_TRUE : GL_FALSE);
}

void GLRenderer::SetPixelPackType(const PixelPackType type)
{
    switch(type) {
        case PixelPackType::PACK:
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            break;
        case PixelPackType::UNPACK:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            break;
    }
}

void GLRenderer::SetBlendInternal(bool enabled)
{
    if(enabled) {
        glEnable(GL_BLEND);
    } else {
        glDisable(GL_BLEND);
    }
}

void GLRenderer::SetBlendFunctionInternal(RendererBlendFunction source, RendererBlendFunction destination)
{
    glBlendFunc(GLUtilities::RendererBlendFunctionToGL(source), GLUtilities::RendererBlendFunctionToGL(destination));
}

void GLRenderer::SetBlendEquationInternal(RendererBlendFunction blendEquation)
{
    CRASH_COND_MSG(false, "Not implemented");
}

void GLRenderer::SetViewportInternal(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    glViewport(x, y, width, height);
}

const String& GLRenderer::GetTitleInternal() const
{
    return m_RendererTitle;
}

void GLRenderer::SetRenderModeInternal(RenderMode mode)
{
}

void GLRenderer::OnResize(uint32_t width, uint32_t height)
{
//    ((GLSwapChain*)Renderer::GetMainSwapChain())->OnResize(width, height);
}

void GLRenderer::SetCullingInternal(bool enabled, bool front)
{
    if(enabled) {
        glEnable(GL_CULL_FACE);
        glCullFace(front ? GL_FRONT : GL_BACK);
    } else {
        glDisable(GL_CULL_FACE);
    }
}

void GLRenderer::SetStencilTestInternal(bool enabled)
{
    if(enabled) {
        glEnable(GL_STENCIL_TEST);
    } else {
        glDisable(GL_STENCIL_TEST);
    }
}

void GLRenderer::SetStencilFunctionInternal(const StencilType type, uint32_t ref, uint32_t mask)
{
    glStencilFunc(GLUtilities::StencilTypeToGL(type), ref, mask);
}

void GLRenderer::SetStencilOpInternal(const StencilType fail, const StencilType zfail, const StencilType zpass)
{
    glStencilOp(GLUtilities::StencilTypeToGL(fail), GLUtilities::StencilTypeToGL(zfail), GLUtilities::StencilTypeToGL(zpass));
}

void GLRenderer::SetColourMaskInternal(bool r, bool g, bool b, bool a)
{
    glColorMask(r, g, b, a);
}

void GLRenderer::DrawInternal(CommandBuffer* commandBuffer, const DrawType type, uint32_t count, DataType dataType, void* indices) const
{
//    Engine::Get().Statistics().NumDrawCalls++;

    if(m_BoundVertexBuffer == -1)
    {
        m_DefaultVertexBuffer->Bind(commandBuffer, (Pipeline*)m_BoundPipeline);
    }

    if(m_BoundIndexBuffer == -1)
    {
        m_DefaultIndexBuffer->Bind(commandBuffer);
    }
    glDrawArrays(GLUtilities::DrawTypeToGL(type), 0, count);

    // GLCall(glDrawElements(GLUtilities::DrawTypeToGL(type), count, GLUtilities::DataTypeToGL(dataType), indices));
}

void GLRenderer::DrawIndexedInternal(CommandBuffer* commandBuffer, const DrawType type, uint32_t count, uint32_t start) const
{
    if(m_BoundIndexBuffer == -1)
    {
        m_DefaultVertexBuffer->Bind(commandBuffer, nullptr);
    }

    if(m_BoundVertexBuffer == -1)
    {
        m_DefaultIndexBuffer->Bind(commandBuffer);
    }

//    Engine::Get().Statistics().NumDrawCalls++;
    glDrawElements(GLUtilities::DrawTypeToGL(type), count, GLUtilities::DataTypeToGL(DataType::UNSIGNED_INT), nullptr);
    // GLCall(glDrawArrays(GLTools::DrawTypeToGL(type), start, count));
}

void GLRenderer::BindDescriptorSetsInternal(Pipeline* pipeline,
                                            CommandBuffer* commandBuffer,
                                            uint32_t dynamicOffset,
                                            DescriptorSet** descriptorSets,
                                            uint32_t descriptorCount)
{
    for(uint32_t i = 0; i < descriptorCount; i++)
    {
        if(descriptorSets[i])
            static_cast<GLDescriptorSet*>(descriptorSets[i])->Bind(dynamicOffset);
    }
}

void GLRenderer::ClearRenderTarget(Texture* texture, CommandBuffer* commandBuffer, Color clearColour)
{
    if(!texture)
    {
        // Assume swapchain texture
        // TODO: function for clearing swapchain image

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    else
    {
        std::vector<TextureType> attachmentTypes = { texture->GetType() };
        std::vector<Texture*> attachments        = { texture };

        //            Graphics::RenderPassDesc renderPassDesc;
        //            renderPassDesc.attachmentCount = uint32_t(attachmentTypes.size());
        //            renderPassDesc.attachmentTypes = attachmentTypes.data();
        //            renderPassDesc.attachments = attachments.data();
        //            renderPassDesc.clear = false;
        //
        //            auto renderPass = Graphics::RenderPass::Get(renderPassDesc);

        glClearColor(static_cast<float>(clearColour.redF()),
                     static_cast<float>(clearColour.greenF()),
                     static_cast<float>(clearColour.blueF()),
                     static_cast<float>(clearColour.alphaF()));

        FramebufferDesc frameBufferDesc {};
        frameBufferDesc.width           = texture->GetWidth();
        frameBufferDesc.height          = texture->GetHeight();
        frameBufferDesc.attachmentCount = uint32_t(attachments.size());
        frameBufferDesc.renderPass      = nullptr;
        frameBufferDesc.attachmentTypes = attachmentTypes.data();
        frameBufferDesc.attachments     = attachments.data();

        auto framebuffer = Framebuffer::Get(frameBufferDesc);
        framebuffer->Bind();
    }
    GLRenderer::ClearInternal(RENDERER_BUFFER_COLOUR | RENDERER_BUFFER_DEPTH | RENDERER_BUFFER_STENCIL);
}

void GLRenderer::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

Renderer* GLRenderer::CreateFuncGL()
{
    return new GLRenderer();
}

}