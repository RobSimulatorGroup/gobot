/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-13
*/

#pragma once

#include <glad/gl.h>

namespace gobot::opengl {

class GLTexture final
{
public:
    GLTexture(GLenum type, const char* file_name);

    GLTexture(GLenum type, const char* file_name, GLenum clamp);

    GLTexture(GLenum type, int width, int height, GLenum internal_format);

    GLTexture(int w, int h, const void* img);

    virtual ~GLTexture();

    GLTexture(const GLTexture&) = delete;

    GLTexture(GLTexture&&);

    GLenum GetType() const { return type_; }

    GLuint GetHandle() const { return handle_; }

    GLuint64 GetHandleBindless() const { return handle_bindless_; }

private:
    GLenum type_ = 0;
    GLuint handle_ = 0;
    GLuint64 handle_bindless_ = 0;
};


}
