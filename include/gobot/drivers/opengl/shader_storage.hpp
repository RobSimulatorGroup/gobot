/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-27
*/

#pragma once

#include "glad/glad.h"
#include "gobot/core/rid_owner.hpp"
#include "gobot/rendering/shader_storage.hpp"
#include "gobot/scene/resources/shader.hpp"

namespace gobot::opengl {

class GOBOT_EXPORT GLShaderStorage : public ShaderStorage {
public:
    GLShaderStorage();

    ~GLShaderStorage() override;

    RID ShaderAllocate() override;

    void ShaderFree(RID p_rid) override;

    bool OwnsShader(RID p_rid) { return shader_owner_.Owns(p_rid); }

    void ShaderInitialize(RID p_shader, ShaderType p_type) override;

    void ShaderSetCode(RID p_shader, const String &p_code) override;

    String ShaderGetCode(RID p_shader) override;

    static GLShaderStorage* GetInstance();

private:
    static GLShaderStorage *s_singleton;


    struct Shader {
        GLuint gl_id;
        String code;
        ShaderType shader_type;
    };

    mutable RID_Owner<Shader, true> shader_owner_;
};

///////////////////////////////////////////////////////

class GOBOT_EXPORT GLShaderProgramStorage : public ShaderProgramStorage {
public:
    GLShaderProgramStorage();

    ~GLShaderProgramStorage() override;

    RID ShaderProgramAllocate() override;

    void ShaderProgramFree(RID p_rid) override;

    bool OwnsShaderProgram(RID p_rid) { return program_owner_.Owns(p_rid); }

    void ShaderProgramInitialize(RID p_shader_program,
                                 const Ref<Shader>& p_vs_shader,
                                 const Ref<Shader>& p_fs_shader,
                                 const Ref<Shader>& p_geometry_shader,
                                 const Ref<Shader>& p_tess_control_shader,
                                 const Ref<Shader>& p_tess_evaluation_shader) override;

    void ShaderProgramInitialize(RID p_shader_program, const Ref<Shader>& p_comp_shader) override;

    static GLShaderProgramStorage* GetInstance();

private:
    static GLShaderProgramStorage *s_singleton;

    struct ShaderProgram {
        GLuint gl_program_id;
        std::vector<RID> shaders;
    };

    mutable RID_Owner<ShaderProgram, true> program_owner_;
};

}
