/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-29
*/

#pragma once

#include "gobot/core/rid.hpp"
#include "gobot/core/types.hpp"
#include "gobot/scene/resources/shader.hpp"

namespace gobot {

class MaterialStorage {
public:
    virtual ~MaterialStorage() = default;

    // shader
    virtual RID ShaderAllocate() = 0;

    virtual void ShaderInitialize(RID p_rid, ShaderType shader_type) = 0;

    virtual void ShaderSetCode(RID p_shader, const String &p_code) = 0;

    virtual String ShaderGetCode(RID p_shader) const = 0;

    virtual void ShaderFree(RID p_rid) = 0;

    // shader program
    virtual RID ShaderProgramAllocate() = 0;

    virtual void ShaderProgramInitialize(RID p_rid) = 0;

    virtual void ShaderProgramFree(RID p_rid) = 0;

    virtual void ShaderProgramSetRasterizerShader(RID p_shader_program,
                                                  RID p_vs_shader,
                                                  RID p_fs_shader,
                                                  RID p_geometry_shader,
                                                  RID p_tess_control_shader,
                                                  RID p_tess_evaluation_shader) = 0;

    virtual void ShaderProgramSetComputeShader(RID p_shader_program, RID p_comp_shader) = 0;

    // material
    virtual RID MaterialAllocate() = 0;

    virtual void MaterialInitialize(RID p_rid) = 0;

    virtual void MaterialFree(RID p_rid) = 0;

};


}
