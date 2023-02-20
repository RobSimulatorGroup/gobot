/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include <gobot_export.h>
#include "gobot/core/types.hpp"
#include "definitions.hpp"
#include "descriptor_set.hpp"

namespace gobot {

class GOBOT_EXPORT Shader
{
public:
    static const Shader* s_CurrentlyBound;

public:
    virtual void Bind() const   = 0;
    virtual void Unbind() const = 0;

    virtual ~Shader() = default;

    virtual const std::vector<ShaderType> GetShaderTypes() const = 0;
    virtual const String& GetName() const                   = 0;
    virtual const String& GetFilePath() const               = 0;

    virtual void* GetHandle() const = 0;
    virtual bool IsCompiled() const { return true; }

    virtual std::vector<PushConstant>& GetPushConstants() = 0;

    virtual PushConstant* GetPushConstant(uint32_t index) { return nullptr; }

    virtual void BindPushConstants(CommandBuffer* commandBuffer, Pipeline* pipeline) = 0;

    virtual DescriptorSetInfo GetDescriptorInfo(uint32_t index) { return DescriptorSetInfo(); }

//    ShaderDataType SPIRVTypeToLumosDataType(const spirv_cross::SPIRType type);

public:
    static Shader* CreateFromFile(const String& filepath);

    static Shader* CreateFromEmbeddedArray(const uint32_t* vertData, uint32_t vertDataSize, const uint32_t* fragData, uint32_t fragDataSize);

    static Shader* CreateCompFromEmbeddedArray(const uint32_t* compData, uint32_t compDataSize);

protected:
    static Shader* (*CreateFunc)(const String&);
    static Shader* (*CreateFuncFromEmbedded)(const uint32_t*, uint32_t, const uint32_t*, uint32_t);
    static Shader* (*CreateCompFuncFromEmbedded)(const uint32_t*, uint32_t);
};

}
