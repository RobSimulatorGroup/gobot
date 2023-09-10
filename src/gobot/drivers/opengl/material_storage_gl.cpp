
/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-29
*/

#include "gobot/drivers/opengl/material_storage_gl.hpp"

namespace gobot::opengl {


inline GLenum ShaderTypeToGLType(ShaderType shader_type) {
    switch (shader_type) {
        case ShaderType::VertexShader:
            return GL_VERTEX_SHADER;
        case ShaderType::TessControlShader:
            return GL_TESS_CONTROL_SHADER;
        case ShaderType::TessEvaluationShader:
            return GL_TESS_EVALUATION_SHADER;
        case ShaderType::GeometryShader:
            return GL_GEOMETRY_SHADER;
        case ShaderType::FragmentShader:
            return GL_FRAGMENT_SHADER;
        case ShaderType::ComputeShader:
            return GL_COMPUTE_SHADER;
        default:
            return GL_NONE;
    }
}

GLMaterialStorage* GLMaterialStorage::s_singleton = nullptr;

GLMaterialStorage *GLMaterialStorage::GetInstance() {
    return s_singleton;
}

GLMaterialStorage::GLMaterialStorage() {
    s_singleton = this;
}

GLMaterialStorage::~GLMaterialStorage() {
    s_singleton = nullptr;
}

RID GLMaterialStorage::ShaderAllocate() {
    return shader_owner_.AllocateRID();
}

void GLMaterialStorage::ShaderInitialize(RID p_rid, ShaderType shader_type) {
    ShaderData shader;
    shader.shader_type = shader_type;
    shader.gl_shader = glCreateShader(ShaderTypeToGLType(shader_type));
    ERR_FAIL_COND(shader.gl_shader == 0);
    shader_owner_.InitializeRID(p_rid, shader);
}

void GLMaterialStorage::ShaderSetCode(RID p_shader, const std::string &p_code, const std::string& p_name, const std::string& p_path) {
    auto* shader = shader_owner_.GetOrNull(p_shader);
    ERR_FAIL_COND(!shader);
    auto str = p_code;
    const char* data = str.data();
    glShaderSource(shader->gl_shader, 1, &data, nullptr);
    glCompileShader(shader->gl_shader);
    GLint status;
    glGetShaderiv(shader->gl_shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLsizei iloglen;
        glGetShaderiv(shader->gl_shader, GL_INFO_LOG_LENGTH, &iloglen);

        if (iloglen < 0) {
            LOG_ERROR("{} compilation failed. No OpenGL shader compiler log.", magic_enum::enum_name(shader->shader_type));
        } else {
            if (iloglen == 0) {
                iloglen = 4096;
            }

            char *ilogmem = (char *)malloc(iloglen + 1);
            ilogmem[iloglen] = '\0';
            glGetShaderInfoLog(shader->gl_shader, iloglen, &iloglen, ilogmem);

            std::string err_string = fmt::format("{} Shader name: {}, path: {}  compilation failed:\n",
                                            magic_enum::enum_name(shader->shader_type), p_name, p_path);

            err_string += ilogmem;
            free(ilogmem);
            LOG_ERROR(err_string);
        }
    }

}

void GLMaterialStorage::ShaderFree(RID p_rid) {
    auto* shader = shader_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(!shader);
    glDeleteShader(shader->gl_shader);
}

RID GLMaterialStorage::ShaderProgramAllocate() {
    return program_owner_.AllocateRID();
}

void GLMaterialStorage::ShaderProgramInitialize(RID p_rid) {
    ShaderProgramData shader_program_data;
    shader_program_data.gl_program = glCreateProgram();
    ERR_FAIL_COND(!shader_program_data.gl_program);
    program_owner_.InitializeRID(p_rid, shader_program_data);
}

void GLMaterialStorage::ShaderProgramFree(RID p_rid) {
    auto* program = program_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(!program);
    glDeleteShader(program->gl_program);
}

void GLMaterialStorage::ShaderProgramSetRasterizerShader(RID p_shader_program,
                                                         RID p_vs_shader,
                                                         RID p_fs_shader,
                                                         RID p_geometry_shader,
                                                         RID p_tess_control_shader,
                                                         RID p_tess_evaluation_shader,
                                                         const std::string& p_name) {
    auto* program = program_owner_.GetOrNull(p_shader_program);
    ERR_FAIL_COND(!program);
    ERR_FAIL_COND(!p_vs_shader.IsValid());
    ERR_FAIL_COND(!p_fs_shader.IsValid());

    auto* vs_shader = shader_owner_.GetOrNull(p_vs_shader);
    auto* fs_shader = shader_owner_.GetOrNull(p_fs_shader);
    ERR_FAIL_COND(!vs_shader);
    ERR_FAIL_COND(!fs_shader);

    glAttachShader(program->gl_program, vs_shader->gl_shader);
    glAttachShader(program->gl_program, fs_shader->gl_shader);

    if (p_geometry_shader.IsValid()) {
        auto* geo_shader = shader_owner_.GetOrNull(p_geometry_shader);
        glAttachShader(program->gl_program, geo_shader->gl_shader);
    }
    if (p_tess_control_shader.IsValid()) {
        auto* tess_control_shader = shader_owner_.GetOrNull(p_tess_control_shader);
        glAttachShader(program->gl_program, tess_control_shader->gl_shader);
    }
    if (p_tess_evaluation_shader.IsValid()) {
        auto* tess_evaluation_shader = shader_owner_.GetOrNull(p_tess_evaluation_shader);
        glAttachShader(program->gl_program, tess_evaluation_shader->gl_shader);
    }

    glLinkProgram(program->gl_program);
    GLint status;
    glGetProgramiv(program->gl_program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLsizei iloglen;
        glGetProgramiv(program->gl_program, GL_INFO_LOG_LENGTH, &iloglen);

        if (iloglen < 0) {
            LOG_ERROR("No OpenGL program link log. Something is wrong.");
            return;
        }

        if (iloglen == 0) {
            iloglen = 4096;
        }

        char *ilogmem = (char *)malloc(iloglen + 1);

        glGetProgramInfoLog(program->gl_program, iloglen, &iloglen, ilogmem);
        ilogmem[iloglen] = '\0';

        std::string err_string = p_name + ": Program linking failed:\n";
        err_string += ilogmem;

        LOG_ERROR(err_string);

        free(ilogmem);
    }
}

void GLMaterialStorage::ShaderProgramSetComputeShader(RID p_shader_program, RID p_comp_shader, const std::string& p_name) {
    auto* program = program_owner_.GetOrNull(p_shader_program);
    ERR_FAIL_COND(!program);
    ERR_FAIL_COND(!p_comp_shader.IsValid());

    auto* comp_shader = shader_owner_.GetOrNull(p_comp_shader);
    ERR_FAIL_COND(!comp_shader);

    glAttachShader(program->gl_program, comp_shader->gl_shader);

    glLinkProgram(program->gl_program);
    GLint status;
    glGetProgramiv(program->gl_program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLsizei iloglen;
        glGetProgramiv(program->gl_program, GL_INFO_LOG_LENGTH, &iloglen);

        if (iloglen < 0) {
            LOG_ERROR("No OpenGL program link log. Something is wrong.");
            return;
        }

        if (iloglen == 0) {
            iloglen = 4096;
        }

        char *ilogmem = (char *)malloc(iloglen + 1);

        glGetProgramInfoLog(program->gl_program, iloglen, &iloglen, ilogmem);
        ilogmem[iloglen] = '\0';

        std::string err_string = p_name + ": Program linking failed:\n";
        err_string += ilogmem;

        LOG_ERROR(err_string);

        free(ilogmem);
    }
}


RID GLMaterialStorage::MaterialAllocate() {
    return RID();
}

void GLMaterialStorage::MaterialInitialize(RID p_rid) {

}

void GLMaterialStorage::MaterialFree(RID p_rid) {

}

std::string GLMaterialStorage::ShaderGetCode(RID p_shader) const {
    return {};
}

}
