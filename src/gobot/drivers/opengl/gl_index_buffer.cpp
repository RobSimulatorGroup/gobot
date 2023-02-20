/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_index_buffer.hpp"
#include "gobot/drivers/opengl/gl.hpp"
#include "gobot/log.hpp"

namespace gobot {

static uint32_t BufferUsageToOpenGL(const BufferUsage usage)
{
    switch(usage)
    {
        case BufferUsage::STATIC:
            return GL_STATIC_DRAW;
        case BufferUsage::DYNAMIC:
            return GL_DYNAMIC_DRAW;
        case BufferUsage::STREAM:
            return GL_STREAM_DRAW;
    }
    return 0;
}

GLIndexBuffer::GLIndexBuffer(uint16_t* data, uint32_t count, BufferUsage bufferUsage)
        : m_Count(count)
        , m_Usage(bufferUsage)
{
    glGenBuffers(1, &m_Handle);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Handle);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(uint16_t), data, BufferUsageToOpenGL(m_Usage));
}

GLIndexBuffer::GLIndexBuffer(uint32_t* data, uint32_t count, BufferUsage bufferUsage)
        : m_Count(count)
        , m_Usage(bufferUsage)
{
    glGenBuffers(1, &m_Handle);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Handle);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(uint32_t), data, BufferUsageToOpenGL(m_Usage));
}

GLIndexBuffer::~GLIndexBuffer()
{
    glDeleteBuffers(1, &m_Handle);
}

void GLIndexBuffer::Bind(CommandBuffer* commandBuffer) const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Handle);
//    GLRenderer::Instance()->GetBoundIndexBuffer() = m_Handle;
}

void GLIndexBuffer::Unbind() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
//    GLRenderer::Instance()->GetBoundIndexBuffer() = -1;
}

uint32_t GLIndexBuffer::GetCount() const
{
    return m_Count;
}

void* GLIndexBuffer::GetPointerInternal()
{
    void* result = nullptr;
    if(!m_Mapped) {
        result = glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
        m_Mapped = true;
    } else
    {
        LOG_WARN("Index buffer already mapped");
    }

    return result;
}

void GLIndexBuffer::ReleasePointer()
{
    if(m_Mapped) {
        glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        m_Mapped = false;
    }
}

void GLIndexBuffer::MakeDefault()
{
    CreateFunc   = CreateFuncGL;
    Create16Func = CreateFunc16GL;
}

IndexBuffer* GLIndexBuffer::CreateFuncGL(uint32_t* data, uint32_t count, BufferUsage bufferUsage)
{
    return new GLIndexBuffer(data, count, bufferUsage);
}

IndexBuffer* GLIndexBuffer::CreateFunc16GL(uint16_t* data, uint32_t count, BufferUsage bufferUsage)
{
    return new GLIndexBuffer(data, count, bufferUsage);
}

}