/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-27
*/

#include "gobot/drivers/opengl/shader_storage.hpp"

namespace gobot::opengl {

GLShaderStorage* GLShaderStorage::s_singleton = nullptr;

GLShaderStorage::GLShaderStorage() {
    s_singleton = this;
}

GLShaderStorage::~GLShaderStorage() {
    s_singleton = nullptr;
}

RID GLShaderStorage::ShaderAllocate() {
    return shader_owner_.AllocateRID();
}

void GLShaderStorage::ShaderFree(RID p_rid) {
    shader_owner_.Free(p_rid);
}

void GLShaderStorage::ShaderInitialize(RID shader_rid, ShaderType p_type) {
    Shader shader;
    // TODO(wqq)
    shader.gl_id = 0; // invalid
    shader.shader_type = ShaderType::None;

    shader_owner_.InitializeRID(shader_rid, shader);
}

void GLShaderStorage::ShaderSetCode(RID shader_id, const String &p_code) {
    Shader *shader = shader_owner_.GetOrNull(shader_id);
    ERR_FAIL_COND(!shader);
    shader->code = p_code;
}


String GLShaderStorage::ShaderGetCode(RID shader_id) {
    Shader *shader = shader_owner_.GetOrNull(shader_id);
    ERR_FAIL_COND_V(!shader, String());
    return shader->code;
}

GLShaderStorage *GLShaderStorage::GetInstance() {
    return s_singleton;
}

//////////////////////////////////////////////////////////////////////

GLShaderProgramStorage* GLShaderProgramStorage::s_singleton = nullptr;

GLShaderProgramStorage::GLShaderProgramStorage() {
    s_singleton = this;
}

GLShaderProgramStorage::~GLShaderProgramStorage() {
    s_singleton = nullptr;
}

RID GLShaderProgramStorage::ShaderProgramAllocate() {
    return program_owner_.AllocateRID();
}

void GLShaderProgramStorage::ShaderProgramFree(RID p_rid) {
    program_owner_.Free(p_rid);
}

void GLShaderProgramStorage::ShaderProgramInitialize(RID p_shader_program, const std::vector<RID>& shaders) {

}


GLShaderProgramStorage* GLShaderProgramStorage::GetInstance() {
    return s_singleton;
}


}
