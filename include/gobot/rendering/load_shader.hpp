/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-25
*/

#pragma once

#include "render_server.hpp"
#include "gobot/core/types.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"
#include <fstream>

namespace gobot {

inline ShaderHandle LoadShader(const char* name)
{
    String shader_path = "???";
    switch (GET_RENDER_SERVER()->GetRendererType())
    {
        case RendererType::Noop:
        case RendererType::Direct3D9: shader_path = "shaders/dx9/";   break;
        case RendererType::Direct3D11:
        case RendererType::Direct3D12: shader_path = "shaders/dx11/";  break;
        case RendererType::Agc:
        case RendererType::Gnm: shader_path = "shaders/pssl/";  break;
        case RendererType::Metal: shader_path = "shaders/metal/"; break;
        case RendererType::Nvn: shader_path = "shaders/nvn/";   break;
        case RendererType::OpenGL: shader_path = "shaders/glsl/";  break;
        case RendererType::OpenGLES: shader_path = "shaders/essl/";  break;
        case RendererType::Vulkan: shader_path = "shaders/spirv/"; break;
        case RendererType::WebGPU: shader_path = "shaders/spirv/"; break;
        case RendererType::Count:
            break;
    }

    shader_path = shader_path + name + ".bin";

    std::ifstream ifs;
    ifs.open(shader_path.toStdString(), std::ios::binary);
    ifs.seekg (0, std::ifstream::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ifstream::beg);
    char * buffer = new char [length];
    ifs.read(buffer,length);
    ifs.close();

    const bgfx::Memory* mem = bgfx::alloc(length + 1);
    memcpy(mem->data, buffer, length);
    mem->data[mem->size-1] = '\0';
    ShaderHandle handle = bgfx::createShader(mem);
    bgfx::setName(handle, name);
    return handle;
}

}