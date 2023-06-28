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

void GLShaderStorage::Initialize(RID shader_rid) {
    Shader shader;
    shader.shader_id = 0; // invalid
    shader.shader_type = ShaderType::None;

    shader_owner_.InitializeRID(shader_rid, shader);
}

void GLShaderStorage::ShaderSetCode(RID shader_id, String code) {
    Shader *shader = shader_owner_.GetOrNull(shader_id);
    ERR_FAIL_COND(!shader);

    shader->code = code;
}


String GLShaderStorage::ShaderGetCode(RID shader_id) {
    Shader *shader = shader_owner_.GetOrNull(shader_id);
    ERR_FAIL_COND_V(!shader, String());
    return shader->code;
}

GLShaderStorage *GLShaderStorage::GetInstance() {
    return s_singleton;
}

void GLShaderStorage::ShaderFree(RID p_rid) {
    Shader *shader = shader_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(!shader);

    shader_owner_.Free(p_rid);
}

}
