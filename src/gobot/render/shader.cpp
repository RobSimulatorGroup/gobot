/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/render/shader.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"

namespace gobot {

Shader* (*Shader::CreateFunc)(const String&)                                                    = nullptr;
Shader* (*Shader::CreateFuncFromEmbedded)(const uint32_t*, uint32_t, const uint32_t*, uint32_t) = nullptr;
Shader* (*Shader::CreateCompFuncFromEmbedded)(const uint32_t*, uint32_t)                        = nullptr;

const Shader* Shader::s_CurrentlyBound = nullptr;

Shader* Shader::CreateFromFile(const String& filepath)
{
    CRASH_COND_MSG(CreateFunc == nullptr, "No Shader Create Function");
    return CreateFunc(filepath);
}

Shader* Shader::CreateFromEmbeddedArray(const uint32_t* vertData, uint32_t vertDataSize, const uint32_t* fragData, uint32_t fragDataSize)
{
    CRASH_COND_MSG(CreateFuncFromEmbedded == nullptr, "No Shader Create Function");
    return CreateFuncFromEmbedded(vertData, vertDataSize, fragData, fragDataSize);
}

Shader* Shader::CreateCompFromEmbeddedArray(const uint32_t* compData, uint32_t compDataSize)
{
    CRASH_COND_MSG(CreateCompFuncFromEmbedded == nullptr, "No Shader Create Function");
    return CreateCompFuncFromEmbedded(compData, compDataSize);
}

//ShaderDataType Shader::SPIRVTypeToLumosDataType(const spirv_cross::SPIRType type)
//{
//    switch(type.basetype)
//    {
//        case spirv_cross::SPIRType::Boolean:
//            return ShaderDataType::BOOL;
//        case spirv_cross::SPIRType::Int:
//            if(type.vecsize == 1)
//                return ShaderDataType::INT;
//            if(type.vecsize == 2)
//                return ShaderDataType::IVEC2;
//            if(type.vecsize == 3)
//                return ShaderDataType::IVEC3;
//            if(type.vecsize == 4)
//                return ShaderDataType::IVEC4;
//
//        case spirv_cross::SPIRType::UInt:
//            return ShaderDataType::UINT;
//        case spirv_cross::SPIRType::Float:
//            if(type.columns == 3)
//                return ShaderDataType::MAT3;
//            if(type.columns == 4)
//                return ShaderDataType::MAT4;
//
//            if(type.vecsize == 1)
//                return ShaderDataType::FLOAT32;
//            if(type.vecsize == 2)
//                return ShaderDataType::VEC2;
//            if(type.vecsize == 3)
//                return ShaderDataType::VEC3;
//            if(type.vecsize == 4)
//                return ShaderDataType::VEC4;
//            break;
//        case spirv_cross::SPIRType::Struct:
//            return ShaderDataType::STRUCT;
//    }
//    LUMOS_LOG_WARN("Unknown spirv type!");
//    return ShaderDataType::NONE;
//}

}
