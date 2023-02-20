/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot_export.h"
#include "gobot/graphics/descriptor_set.hpp"
#include "gobot/graphics/uniform_buffer.hpp"

namespace gobot {

class GLShader;

class GLDescriptorSet : public DescriptorSet
{
public:
    GLDescriptorSet(const DescriptorDesc& descriptorDesc);

    ~GLDescriptorSet() {};

    void Update() override;
    void SetTexture(const String& name, Texture* texture, uint32_t mipIndex, TextureType textureType) override;
    void SetTexture(const String& name, Texture** texture, uint32_t textureCount, TextureType textureType) override;
    void SetBuffer(const String& name, UniformBuffer* buffer) override;
    void SetUniform(const String& bufferName, const String& uniformName, void* data) override;
    void SetUniform(const String& bufferName, const String& uniformName, void* data, uint32_t size) override;
    void SetUniformBufferData(const String& bufferName, void* data) override;

    UniformBuffer* GetUnifromBuffer(const String& name) override;
    void Bind(uint32_t offset = 0);

    void SetDynamicOffset(uint32_t offset) override { m_DynamicOffset = offset; }
    uint32_t GetDynamicOffset() const override { return m_DynamicOffset; }
    static void MakeDefault();

protected:
    static DescriptorSet* CreateFuncGL(const DescriptorDesc& descriptorDesc);

private:
    uint32_t m_DynamicOffset = 0;
    GLShader* m_Shader       = nullptr;

    std::vector<Descriptor> m_Descriptors;
    struct UniformBufferInfo
    {
        Ref<UniformBuffer> UB;
        std::vector<BufferMemberInfo> m_Members;
//        Buffer LocalStorage;
        bool HasUpdated;
    };
    std::unordered_map<String, UniformBufferInfo> m_UniformBuffers;
};

}