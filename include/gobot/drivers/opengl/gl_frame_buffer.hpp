/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-13
*/

#pragma once

#include <memory>

#include <glad/gl.h>
#include "gl_texture.hpp"

namespace gobot::opengl {

class GLFramebuffer final {
public:
    GLFramebuffer(int width, int height, GLenum format_color, GLenum format_depth);

    ~GLFramebuffer();

    GLFramebuffer(const GLFramebuffer&) = delete;

    GLFramebuffer(GLFramebuffer&&) = default;

    GLuint GetHandle() const { return handle_; }

    const GLTexture& GetTextureColor() const { return *tex_color_; }

    const GLTexture& GetTextureDepth() const { return *tex_depth_; }

    void Bind();

    void Unbind();

private:
    int width_;
    int height_;
    GLuint handle_{0};

    std::unique_ptr<GLTexture> tex_color_;
    std::unique_ptr<GLTexture> tex_depth_;
};


}
