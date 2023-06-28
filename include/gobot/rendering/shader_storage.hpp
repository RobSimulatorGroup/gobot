/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-27
*/

#pragma once

#include "gobot/scene/resources/shader.hpp"

namespace gobot {

class ShaderStorage {
public:
    virtual ~ShaderStorage() {};

    virtual RID ShaderAllocate() = 0;

    virtual void ShaderFree(RID p_rid) = 0;

    virtual void ShaderInitialize(RID p_shader, ShaderType p_type) = 0;

    virtual void ShaderSetCode(RID p_shader, const String &p_code) = 0;

    virtual String ShaderGetCode(RID p_shader) = 0;
};

class ShaderProgramStorage {
public:
    virtual ~ShaderProgramStorage() {};

    virtual RID ShaderProgramAllocate() = 0;

    virtual void ShaderProgramFree(RID p_rid) = 0;

    virtual void ShaderProgramInitialize(RID p_shader_program, const std::vector<RID>& shaders) = 0;
};


}
