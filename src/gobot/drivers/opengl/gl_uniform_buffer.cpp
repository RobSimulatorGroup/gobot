/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_uniform_buffer.hpp"
#include "gobot/drivers/opengl/gl.hpp"
#include "gobot/drivers/opengl/gl_shader.hpp"

namespace gobot {

GLUniformBuffer::GLUniformBuffer()
{
    glGenBuffers(1, &m_Handle);
}

GLUniformBuffer::~GLUniformBuffer()
{
    glDeleteBuffers(1, &m_Handle);
}

void GLUniformBuffer::Init(uint32_t size, const void* data)
{
    m_Data = (uint8_t*)data;
    m_Size = size;
    glBindBuffer(GL_UNIFORM_BUFFER, m_Handle);
    glBufferData(GL_UNIFORM_BUFFER, m_Size, m_Data, GL_DYNAMIC_DRAW);
}

void GLUniformBuffer::SetData(uint32_t size, const void* data)
{
    m_Data    = (uint8_t*)data;
    GLvoid* p = nullptr;

    glBindBuffer(GL_UNIFORM_BUFFER, m_Handle);

    if(size != m_Size)
    {
        p      = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
        m_Size = size;

        memcpy(p, m_Data, m_Size);
        glUnmapBuffer(GL_UNIFORM_BUFFER);
    }
    else
    {
        glBufferSubData(GL_UNIFORM_BUFFER, 0, m_Size, m_Data);
    }
}

void GLUniformBuffer::SetDynamicData(uint32_t size, uint32_t typeSize, const void* data)
{
    m_Data            = (uint8_t*)data;
    m_Size            = size;
    m_Dynamic         = true;
    m_DynamicTypeSize = typeSize;

    GLvoid* p = nullptr;

    glBindBuffer(GL_UNIFORM_BUFFER, m_Handle);

    if(size != m_Size)
    {
        p      = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
        m_Size = size;

        memcpy(p, m_Data, m_Size);
        glUnmapBuffer(GL_UNIFORM_BUFFER);
    }
    else
    {
        glBufferSubData(GL_UNIFORM_BUFFER, 0, m_Size, m_Data);
    }
}

void GLUniformBuffer::Bind(uint32_t slot, GLShader* shader, String & name)
{
    glBindBufferBase(GL_UNIFORM_BUFFER, slot, m_Handle);
    shader->BindUniformBuffer(this, slot, name);
    // uint32_t location = glGetUniformBlockIndex(shader->GetHandle(), name.c_str());
    // GLCall(glUniformBlockBinding(shader->GetHandle(), location, slot));
}

void GLUniformBuffer::MakeDefault()
{
    CreateFunc     = CreateFuncGL;
    CreateDataFunc = CreateDataFuncGL;
}

UniformBuffer* GLUniformBuffer::CreateDataFuncGL(uint32_t size, const void* data)
{
    // TODO
    return new GLUniformBuffer();
}

UniformBuffer* GLUniformBuffer::CreateFuncGL()
{
    return new GLUniformBuffer();
}

}