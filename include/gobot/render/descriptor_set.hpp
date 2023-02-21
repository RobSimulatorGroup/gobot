/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "definitions.hpp"
#include "gobot_export.h"

namespace gobot {

struct DescriptorSetInfo
{
    std::vector<Descriptor> descriptors;
};

class GOBOT_EXPORT DescriptorSet
{
public:
    virtual ~DescriptorSet() = default;
    static DescriptorSet* Create(const DescriptorDesc& desc);

    virtual void Update() = 0;
    virtual void SetDynamicOffset(uint32_t offset) = 0;
    virtual uint32_t GetDynamicOffset() const = 0;
    virtual void SetTexture(const String& name, Texture** texture, uint32_t textureCount, TextureType textureType = TextureType(0)) = 0;
    virtual void SetTexture(const String& name, Texture* texture, uint32_t mipIndex = 0, TextureType textureType = TextureType(0))  = 0;
    virtual void SetBuffer(const String& name, UniformBuffer* buffer) = 0;
    virtual UniformBuffer* GetUnifromBuffer(const String& name) = 0;
    virtual void SetUniform(const String& bufferName, const String& uniformName, void* data) = 0;
    virtual void SetUniform(const String& bufferName, const String& uniformName, void* data, uint32_t size) = 0;
    virtual void SetUniformBufferData(const String& bufferName, void* data) = 0;
    virtual void TransitionImages(CommandBuffer* commandBuffer = nullptr) { }
    virtual void SetUniformDynamic(const String& bufferName, uint32_t size) { }

protected:
    static DescriptorSet* (*CreateFunc)(const DescriptorDesc&);
};

}
