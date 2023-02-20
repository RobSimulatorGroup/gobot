/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_vertex_buffer.hpp"
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

GLVertexBuffer::GLVertexBuffer(BufferUsage usage)
        : m_Usage(usage)
        , m_Size(0)
{
    glGenBuffers(1, &m_Handle);
}

GLVertexBuffer::~GLVertexBuffer()
{
    glDeleteBuffers(1, &m_Handle);
}

void GLVertexBuffer::Resize(uint32_t size)
{
    m_Size = size;

    glBindBuffer(GL_ARRAY_BUFFER, m_Handle);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, BufferUsageToOpenGL(m_Usage));
}

void GLVertexBuffer::SetData(uint32_t size, const void* data)
{
    m_Size = size;
    glBindBuffer(GL_ARRAY_BUFFER, m_Handle);
    glBufferData(GL_ARRAY_BUFFER, size, data, BufferUsageToOpenGL(m_Usage));
}

void GLVertexBuffer::SetDataSub(uint32_t size, const void* data, uint32_t offset)
{
    m_Size = size;
    glBindBuffer(GL_ARRAY_BUFFER, m_Handle);
    glBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
}

void* GLVertexBuffer::GetPointerInternal()
{
    void* result = nullptr;
    if(!m_Mapped)
    {
        result = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        m_Mapped = true;
    }
    else
    {
        LOG_WARN("Vertex buffer already mapped");
    }

    return result;
}

void GLVertexBuffer::ReleasePointer()
{
    if(m_Mapped)
    {
        glUnmapBuffer(GL_ARRAY_BUFFER);
        m_Mapped = false;
    }
}

void GLVertexBuffer::Bind(CommandBuffer* commandBuffer, Pipeline* pipeline)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_Handle);
//    GLRenderer::Instance()->GetBoundVertexBuffer() = m_Handle;
//
//    if(pipeline)
//        ((GLPipeline*)pipeline)->BindVertexArray();
}

void GLVertexBuffer::Unbind()
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
//    GLRenderer::Instance()->GetBoundVertexBuffer() = -1;
}

void GLVertexBuffer::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

VertexBuffer* GLVertexBuffer::CreateFuncGL(const BufferUsage& usage)
{
    return new GLVertexBuffer(usage);
}

}