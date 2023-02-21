/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_command_buffer.hpp"
#include "gobot/render/pipeline.hpp"

namespace gobot {

GLCommandBuffer::GLCommandBuffer()
        : primary(false)
        , m_BoundPipeline(nullptr)
{
}

GLCommandBuffer::~GLCommandBuffer()
{
}

bool GLCommandBuffer::Init(bool primary)
{
    return true;
}

void GLCommandBuffer::Unload()
{
}

void GLCommandBuffer::BeginRecording()
{
}

void GLCommandBuffer::BeginRecordingSecondary(RenderPass* renderPass, Framebuffer* framebuffer)
{
}

void GLCommandBuffer::EndRecording()
{
}

void GLCommandBuffer::ExecuteSecondary(CommandBuffer* primaryCmdBuffer)
{
}

void GLCommandBuffer::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

void GLCommandBuffer::BindPipeline(Pipeline* pipeline)
{
    if(pipeline != m_BoundPipeline)
    {
        pipeline->Bind(this);
        m_BoundPipeline = pipeline;
    }
}

void GLCommandBuffer::UnBindPipeline()
{
    if(m_BoundPipeline)
        m_BoundPipeline->End(this);
    m_BoundPipeline = nullptr;
}

CommandBuffer* GLCommandBuffer::CreateFuncGL()
{
    return new GLCommandBuffer();
}

}