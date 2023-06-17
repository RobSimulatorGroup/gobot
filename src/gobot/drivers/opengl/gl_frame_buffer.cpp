/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-13
*/

#include <cassert>
#include "gobot/drivers/opengl/gl_frame_buffer.hpp"

namespace gobot::opengl {

GLFramebuffer::GLFramebuffer(int width, int height, GLenum format_color, GLenum format_depth)
    : width_(width),
      height_(height)
{
    glCreateFramebuffers(1, &handle_);

    if (format_color) {
        tex_color_ = std::make_unique<GLTexture>(GL_TEXTURE_2D, width, height, format_color);
        glTextureParameteri(tex_color_->GetHandle(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(tex_color_->GetHandle(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glNamedFramebufferTexture(handle_, GL_COLOR_ATTACHMENT0, tex_color_->GetHandle(), 0);
    }

    if (format_depth) {
        tex_depth_ = std::make_unique<GLTexture>(GL_TEXTURE_2D, width, height, format_depth);
        const GLfloat border[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        glTextureParameterfv(tex_depth_->GetHandle(), GL_TEXTURE_BORDER_COLOR, border);
        glTextureParameteri(tex_depth_->GetHandle(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTextureParameteri(tex_depth_->GetHandle(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glNamedFramebufferTexture(handle_, GL_DEPTH_ATTACHMENT, tex_depth_->GetHandle(), 0);
    }

    const GLenum status = glCheckNamedFramebufferStatus(handle_, GL_FRAMEBUFFER);

    assert(status == GL_FRAMEBUFFER_COMPLETE);
}

GLFramebuffer::~GLFramebuffer() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &handle_);
}


}
