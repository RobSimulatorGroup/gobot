/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_frame_buffer.hpp"
#include "gobot/drivers/opengl/gl_texture.hpp"
#include "gobot/log.hpp"

namespace gobot {

GLFramebuffer::GLFramebuffer()
        : m_Width(0)
        , m_Height(0)
{
    glGenFramebuffers(1, &m_Handle);
    glBindFramebuffer(GL_FRAMEBUFFER, m_Handle);
    m_ColourAttachmentCount = 0;
}

GLFramebuffer::GLFramebuffer(const FramebufferDesc& frameBufferDesc)
{
    m_ScreenFramebuffer     = frameBufferDesc.screenFBO;
    m_Width                 = frameBufferDesc.width;
    m_Height                = frameBufferDesc.height;
    m_ColourAttachmentCount = 0;

    if(m_ScreenFramebuffer) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {
        glGenFramebuffers(1, &m_Handle);
        glBindFramebuffer(GL_FRAMEBUFFER, m_Handle);

        for(uint32_t i = 0; i < frameBufferDesc.attachmentCount; i++) {
            switch(frameBufferDesc.attachmentTypes[i])
            {
                case TextureType::COLOUR:
                    AddTextureAttachment(frameBufferDesc.attachments[i]->GetFormat(), frameBufferDesc.attachments[i]);
                    break;
                case TextureType::DEPTH:
                    AddTextureAttachment(RHIFormat::D16_Unorm, frameBufferDesc.attachments[i]);
                    break;
                case TextureType::DEPTHARRAY:
                    AddTextureLayer(frameBufferDesc.layer, frameBufferDesc.attachments[i]);
                    break;
                case TextureType::OTHER: {
                    LOG_ERROR("Unimplemented");
                    break;
                }
                case TextureType::CUBE: {
                    LOG_ERROR("Unimplemented");
                    break;
                }
            }
        }

        glDrawBuffers(static_cast<GLsizei>(m_AttachmentData.size()), m_AttachmentData.data());

        Validate();

        UnBind();
    }
}

GLFramebuffer::~GLFramebuffer()
{
    if(!m_ScreenFramebuffer)
        glDeleteFramebuffers(1, &m_Handle);
}

void GLFramebuffer::GenerateFramebuffer()
{
    if(!m_ScreenFramebuffer)
        glGenFramebuffers(1, &m_Handle);
}

void GLFramebuffer::Bind(uint32_t width, uint32_t height) const
{
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_ScreenFramebuffer ? 0 : m_Handle);
    glViewport(0, 0, width, height);

    if(!m_ScreenFramebuffer)
        glDrawBuffers(static_cast<GLsizei>(m_AttachmentData.size()), m_AttachmentData.data());
}

void GLFramebuffer::UnBind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLenum GLFramebuffer::GetAttachmentPoint(RHIFormat format)
{
    if(Texture::IsDepthStencilFormat(format)) {
        return GL_DEPTH_STENCIL_ATTACHMENT;
    }

    if(Texture::IsStencilFormat(format)) {
        return GL_STENCIL_ATTACHMENT;
    }

    if(Texture::IsDepthFormat(format)) {
        return GL_DEPTH_ATTACHMENT;
    }

    GLenum value = GL_COLOR_ATTACHMENT0 + m_ColourAttachmentCount;
    m_ColourAttachmentCount++;
    return value;
}

void GLFramebuffer::Bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_ScreenFramebuffer ? 0 : m_Handle);
}

void GLFramebuffer::AddTextureAttachment(const RHIFormat format, Texture* texture)
{
    GLenum attachment = GetAttachmentPoint(format);

    if(attachment != GL_DEPTH_ATTACHMENT && attachment != GL_STENCIL_ATTACHMENT && attachment != GL_DEPTH_STENCIL_ATTACHMENT)
    {
        m_AttachmentData.emplace_back(attachment);
    }
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment, GL_TEXTURE_2D, (GLuint)(size_t)((GLTexture2D*)texture)->GetHandle(), 0);
}

void GLFramebuffer::AddCubeTextureAttachment(const RHIFormat format, const CubeFace face, TextureCube* texture)
{
    uint32_t faceID = 0;

    GLenum attachment = GetAttachmentPoint(format);
    if(attachment != GL_DEPTH_ATTACHMENT && attachment != GL_STENCIL_ATTACHMENT && attachment != GL_DEPTH_STENCIL_ATTACHMENT)
    {
        m_AttachmentData.emplace_back(attachment);
    }

    switch(face)
    {
        case CubeFace::PositiveX:
            faceID = GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            break;
        case CubeFace::NegativeX:
            faceID = GL_TEXTURE_CUBE_MAP_NEGATIVE_X;
            break;
        case CubeFace::PositiveY:
            faceID = GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
            break;
        case CubeFace::NegativeY:
            faceID = GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;
            break;
        case CubeFace::PositiveZ:
            faceID = GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
            break;
        case CubeFace::NegativeZ:
            faceID = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
            break;
    }

    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, faceID, (GLuint)(size_t)texture->GetHandle(), 0);
}

void GLFramebuffer::AddShadowAttachment(Texture* texture)
{
    glFramebufferTextureLayer(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, (GLuint)(size_t)texture->GetHandle(), 0, 0);
    glDrawBuffers(0, GL_NONE);
}

void GLFramebuffer::AddTextureLayer(int index, Texture* texture)
{
    glFramebufferTextureLayer(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, (GLuint)(size_t)texture->GetHandle(), 0, index);
}

void GLFramebuffer::Validate()
{
    uint32_t status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE)
    {
        LOG_FATAL("Unable to create Framebuffer! StatusCode: {0}", status);
    }
}

void GLFramebuffer::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

Framebuffer* GLFramebuffer::CreateFuncGL(const FramebufferDesc& frameBufferDesc)
{
    return new GLFramebuffer(frameBufferDesc);
}

}