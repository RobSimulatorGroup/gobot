
/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-29
*/

#include "gobot/drivers/opengl/material_storage_gl.hpp"

namespace gobot::opengl {


GLMaterialStorage* GLMaterialStorage::s_singleton = nullptr;

GLMaterialStorage::GLMaterialStorage() {
    s_singleton = this;
}

GLMaterialStorage::~GLMaterialStorage() {
    s_singleton = nullptr;
}

RID GLMaterialStorage::ShaderAllocate() {
    return RID();
}

void GLMaterialStorage::ShaderInitialize(RID p_rid, ShaderType shader_type) {

}

void GLMaterialStorage::ShaderSetCode(RID p_shader, const String &p_code) {

}

void GLMaterialStorage::ShaderFree(RID p_rid) {

}

RID GLMaterialStorage::MaterialAllocate() {
    return RID();
}

void GLMaterialStorage::MaterialInitialize(RID p_rid) {

}

void GLMaterialStorage::MaterialFree(RID p_rid) {

}

GLMaterialStorage *GLMaterialStorage::GetInstance() {
    return s_singleton;
}

String GLMaterialStorage::ShaderGetCode(RID p_shader) const {
    return {};
}

RID GLMaterialStorage::ShaderProgramAllocate() {
    return RID();
}

void GLMaterialStorage::ShaderProgramInitialize(RID p_rid) {

}

void GLMaterialStorage::ShaderProgramFree(RID p_rid) {

}

void GLMaterialStorage::ShaderProgramSetComputeShader(RID p_shader_program, RID p_comp_shader) {

}

void GLMaterialStorage::ShaderProgramSetRasterizerShader(RID p_shader_program,
                                                         RID p_vs_shader,
                                                         RID p_fs_shader,
                                                         RID p_geometry_shader,
                                                         RID p_tess_control_shader,
                                                         RID p_tess_evaluation_shader) {

}


}
