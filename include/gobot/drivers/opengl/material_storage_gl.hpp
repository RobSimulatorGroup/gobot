
/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-29
*/

#pragma once

#include "glad/glad.h"
#include "gobot/rendering/material_storage.hpp"
#include "gobot/core/rid_owner.hpp"

namespace gobot::opengl {

class GLMaterialStorage : public MaterialStorage {
public:
    GLMaterialStorage();

    ~GLMaterialStorage() override;

    // shader
    RID ShaderAllocate() override;

    void ShaderInitialize(RID p_rid, ShaderType shader_type) override;

    bool OwnsShader(RID p_rid) { return shader_owner_.Owns(p_rid); }

    void ShaderSetCode(RID p_shader, const String &p_code) override;

    String ShaderGetCode(RID p_shader) const override;

    void ShaderFree(RID p_rid) override;

    // shader program
    RID ShaderProgramAllocate() override;

    void ShaderProgramInitialize(RID p_rid) override;

    void ShaderProgramFree(RID p_rid) override;

    bool OwnsShaderProgram(RID p_rid) { return program_owner_.Owns(p_rid); }

    void ShaderProgramSetRasterizerShader(RID p_shader_program,
                                          RID p_vs_shader,
                                          RID p_fs_shader,
                                          RID p_geometry_shader,
                                          RID p_tess_control_shader,
                                          RID p_tess_evaluation_shader) override;

    void ShaderProgramSetComputeShader(RID p_shader_program, RID p_comp_shader) override;

    // material
    RID MaterialAllocate() override;

    void MaterialInitialize(RID p_rid) override;

    bool OwnsMaterial(RID p_rid) { return material_owner_.Owns(p_rid); }

    void MaterialFree(RID p_rid) override;

    static GLMaterialStorage* GetInstance();

private:
    static GLMaterialStorage* s_singleton;

    struct ShaderData {
        GLuint gl_shader;
        String code;
        ShaderType shader_type;
    };

    mutable RID_Owner<ShaderData, true> shader_owner_;

    struct ShaderProgramData {
        GLuint gl_program;
        std::vector<RID> shaders;
    };

    mutable RID_Owner<ShaderProgramData, true> program_owner_;


    struct MaterialData {

    };

    mutable RID_Owner<MaterialData, true> material_owner_;

};


}
