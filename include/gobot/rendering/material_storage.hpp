/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-29
 * SPDX-License-Identifier: Apache-2.0
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

    virtual void ShaderSetCode(RID p_shader, const std::string &p_code, const std::string& p_name, const std::string& p_path) = 0;

    virtual std::string ShaderGetCode(RID p_shader) const = 0;

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
                                                  RID p_tess_evaluation_shader,
                                                  const std::string& p_name) = 0;

    virtual void ShaderProgramSetComputeShader(RID p_shader_program, RID p_comp_shader, const std::string& p_name) = 0;

    // material
    virtual RID MaterialAllocate() = 0;

    virtual void MaterialInitialize(RID p_rid) = 0;

    virtual void MaterialFree(RID p_rid) = 0;

};


}
