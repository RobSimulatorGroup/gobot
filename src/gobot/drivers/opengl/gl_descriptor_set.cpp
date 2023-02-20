/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_descriptor_set.hpp"
#include "gobot/drivers/opengl/gl_texture.hpp"
#include "gobot/drivers/opengl/gl_uniform_buffer.hpp"
#include "gobot/drivers/opengl/gl_shader.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

GLDescriptorSet::GLDescriptorSet(const DescriptorDesc& descriptorDesc)
{
    m_Shader      = (GLShader*)descriptorDesc.shader;
    m_Descriptors = m_Shader->GetDescriptorInfo(descriptorDesc.layoutIndex).descriptors;

    for(auto& descriptor : m_Descriptors)
    {
        if(descriptor.type == DescriptorType::UNIFORM_BUFFER)
        {
            auto buffer = Ref<UniformBuffer>(UniformBuffer::Create());
            buffer->Init(descriptor.size, nullptr);
            descriptor.buffer = buffer.Get();

//            Buffer localStorage;
//            localStorage.Allocate(descriptor.size);
//            localStorage.InitialiseEmpty();

            UniformBufferInfo info;
            info.UB           = buffer;
//            info.LocalStorage = localStorage;
            info.HasUpdated   = false;
            info.m_Members    = descriptor.m_Members;
            m_UniformBuffers.emplace(descriptor.name, info);
        }
    }
}

void GLDescriptorSet::Update()
{
    for(auto& bufferInfo : m_UniformBuffers)
    {
        if(bufferInfo.second.HasUpdated)
        {
//            bufferInfo.second.UB->SetData(bufferInfo.second.LocalStorage.Data);
            bufferInfo.second.HasUpdated = false;
        }
    }
}

void GLDescriptorSet::SetTexture(const String& name, Texture* texture, uint32_t mipIndex, TextureType textureType)
{
    for(auto& descriptor : m_Descriptors)
    {
        if(descriptor.type == DescriptorType::IMAGE_SAMPLER && descriptor.name == name)
        {
            descriptor.texture      = texture;
            descriptor.textureType  = textureType;
            descriptor.textureCount = 1;
            return;
        }
    }
    LOG_WARN("Texture not found {0}", name);
}

void GLDescriptorSet::SetTexture(const String& name, Texture** texture, uint32_t textureCount, TextureType textureType)
{
    for(auto& descriptor : m_Descriptors)
    {
        if(descriptor.type == DescriptorType::IMAGE_SAMPLER && descriptor.name == name)
        {
            descriptor.textureCount = textureCount;
            descriptor.textures     = texture;
            descriptor.textureType  = textureType;
            return;
        }
    }
    LOG_WARN("Texture not found {0}", name);
}

void GLDescriptorSet::SetBuffer(const String& name, UniformBuffer* buffer)
{
    // TODO: Remove
    for(auto& descriptor : m_Descriptors)
    {
        if(descriptor.type == DescriptorType::UNIFORM_BUFFER && descriptor.name == name)
        {
            descriptor.buffer = buffer;
            return;
        }
    }

    LOG_WARN("Buffer not found {0}", name);
}

void GLDescriptorSet::SetUniform(const String& bufferName, const String& uniformName, void* data)
{
    auto itr = m_UniformBuffers.find(bufferName);
    if(itr != m_UniformBuffers.end())
    {
        for(auto& member : itr->second.m_Members)
        {
            if(member.name == uniformName)
            {
//                itr->second.LocalStorage.Write(data, member.size, member.offset);
                itr->second.HasUpdated = true;
                return;
            }
        }
    }

    LOG_WARN("Uniform not found {0}.{1}", bufferName, uniformName);
}

void GLDescriptorSet::SetUniform(const String& bufferName, const String& uniformName, void* data, uint32_t size)
{
    auto itr = m_UniformBuffers.find(bufferName);
    if(itr != m_UniformBuffers.end())
    {
        for(auto& member : itr->second.m_Members)
        {
            if(member.name == uniformName)
            {
//                itr->second.LocalStorage.Write(data, size, member.offset);
                itr->second.HasUpdated = true;
                return;
            }
        }
    }

    LOG_WARN("Uniform not found {0}.{1}", bufferName, uniformName);
}

void GLDescriptorSet::SetUniformBufferData(const String& bufferName, void* data)
{
    auto itr = m_UniformBuffers.find(bufferName);
    if(itr != m_UniformBuffers.end())
    {
//        itr->second.LocalStorage.Write(data, itr->second.LocalStorage.GetSize(), 0);
        itr->second.HasUpdated = true;
        return;
    }

    LOG_WARN("Uniform not found {0}.{1}", bufferName);
}

UniformBuffer* GLDescriptorSet::GetUnifromBuffer(const String& name)
{
    for(auto& descriptor : m_Descriptors)
    {
        if(descriptor.type == DescriptorType::UNIFORM_BUFFER && descriptor.name == name)
        {
            return descriptor.buffer;
        }
    }

    LOG_WARN("Buffer not found {0}", name);
    return nullptr;
}

void GLDescriptorSet::Bind(uint32_t offset)
{
    m_Shader->Bind();

    for(auto& descriptor : m_Descriptors)
    {
        if(descriptor.type == DescriptorType::IMAGE_SAMPLER)
        {
            if(descriptor.textureCount == 1)
            {
                if(descriptor.texture)
                {
                    descriptor.texture->Bind(descriptor.binding);
                    m_Shader->SetUniform1i(descriptor.name, descriptor.binding);
                }
            }
            else
            {
                static const int MAX_TEXTURE_UNITS = 16;
                int32_t samplers[MAX_TEXTURE_UNITS];

                CRASH_COND_MSG(MAX_TEXTURE_UNITS >= descriptor.textureCount, "Texture Count greater than max");

                for(uint32_t i = 0; i < descriptor.textureCount; i++)
                {
                    if(descriptor.textures && descriptor.textures[i])
                    {
                        descriptor.textures[i]->Bind(descriptor.binding + i);
                        samplers[i] = i;
                    }
                }
                m_Shader->SetUniform1iv(descriptor.name, samplers, descriptor.textureCount);
            }
        }
        else
        {
            auto* buffer = dynamic_cast<GLUniformBuffer*>(descriptor.buffer);

            if(!buffer)
                break;

            uint8_t* data;
            uint32_t size;

            if(buffer->GetDynamic())
            {
                data = reinterpret_cast<uint8_t*>(buffer->GetBuffer()) + offset;
                size = buffer->GetTypeSize();
            }
            else
            {
                data = buffer->GetBuffer();
                size = buffer->GetSize();
            }

            {
                // buffer->SetData(size, data);
                auto bufferHandle = static_cast<GLUniformBuffer*>(buffer)->GetHandle();
                auto slot         = descriptor.binding;
                {
                    glBindBufferBase(GL_UNIFORM_BUFFER, slot, bufferHandle);
                }

                if(buffer->GetDynamic())
                {
                    glBindBufferRange(GL_UNIFORM_BUFFER, slot, bufferHandle, offset, size);
                }

                if(descriptor.name != "")
                {
                    auto loc = glGetUniformBlockIndex(m_Shader->GetHandleInternal(), descriptor.name.toStdString().c_str());
                    glUniformBlockBinding(m_Shader->GetHandleInternal(), loc, slot);
                }
            }
        }
    }
}

void GLDescriptorSet::MakeDefault()
{
    CreateFunc = CreateFuncGL;
}

DescriptorSet* GLDescriptorSet::CreateFuncGL(const DescriptorDesc& descriptorDesc)
{
    return new GLDescriptorSet(descriptorDesc);
}

}