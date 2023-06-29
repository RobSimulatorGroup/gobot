/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-12
*/

#include "gobot/scene/resources/shader.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Shader::Shader() {
}

Shader::~Shader() {
    RS::GetInstance()->Free(shader_);
}

void Shader::SetCode(const String &p_code) {
    ERR_FAIL_COND_MSG(shader_.IsNull(), "The shader must valid before set code.");
    code_ = p_code;
    RS::GetInstance()->ShaderSetCode(shader_, p_code);
}

String Shader::GetCode() const {
    return code_;
}

void Shader::SetShaderType(ShaderType p_shader_type) {
    if (shader_.IsNull()) {
        shader_ = RS::GetInstance()->ShaderCreate(p_shader_type);
    } else {
        RS::GetInstance()->Free(shader_);
        shader_ = RS::GetInstance()->ShaderCreate(p_shader_type);
        code_.clear();
    }
    shader_type_ = p_shader_type;
}

ShaderType Shader::GetShaderType() const {
    return shader_type_;
}

RID Shader::GetRid() const {
    return shader_;
}

/////////////////////////////////////

RasterizerShaderProgram::RasterizerShaderProgram() {
}

RasterizerShaderProgram::~RasterizerShaderProgram() {
    RS::GetInstance()->Free(shader_program_);
}

void RasterizerShaderProgram::SetRasterizerShader(const Ref<Shader>& p_vs_shader,
                                                  const Ref<Shader>& p_fs_shader,
                                                  const Ref<Shader>& p_geometry_shader,
                                                  const Ref<Shader>& p_tess_control_shader,
                                                  const Ref<Shader>& p_tess_evaluation_shader) {
    ERR_FAIL_COND_MSG(!p_vs_shader.IsValid(), "p_vs_shader must be valid");
    ERR_FAIL_COND_MSG(!p_fs_shader.IsValid(), "p_fs_shader must be valid");

    vertex_shaders_ = p_vs_shader;
    fragment_shader_ = p_fs_shader;
    geometry_shader_ = p_geometry_shader;
    tess_control_shader_ = p_tess_control_shader;
    tess_evaluation_shader_ = p_tess_evaluation_shader;

    if (shader_program_.IsNull()) {
        shader_program_ = RS::GetInstance()->ShaderProgramCreate(vertex_shaders_,
                                                                 fragment_shader_,
                                                                 geometry_shader_,
                                                                 tess_control_shader_,
                                                                 tess_evaluation_shader_);
    } else {
        RS::GetInstance()->Free(shader_program_);
        shader_program_ = RS::GetInstance()->ShaderProgramCreate(vertex_shaders_,
                                                                 fragment_shader_,
                                                                 geometry_shader_,
                                                                 tess_control_shader_,
                                                                 tess_evaluation_shader_);
    }
}

bool RasterizerShaderProgram::IsComplete() {
    return vertex_shaders_.IsValid() && fragment_shader_.IsValid();
}


RID RasterizerShaderProgram::GetRid() const {
    return shader_program_;
}

/////////////////////////////////////////////////////////

ComputeShaderProgram::ComputeShaderProgram() {
}

ComputeShaderProgram::~ComputeShaderProgram() {
    RS::GetInstance()->Free(shader_program_);
}

void ComputeShaderProgram::SetComputeShader(const Ref<Shader>& p_comp_shader) {
    if (shader_program_.IsNull()) {
        shader_program_ = RS::GetInstance()->ShaderProgramCreate(p_comp_shader);
    } else {
        RS::GetInstance()->Free(shader_program_);
        shader_program_ = RS::GetInstance()->ShaderProgramCreate(p_comp_shader);
    }
}


RID ComputeShaderProgram::GetRid() const {
    return shader_program_;
}

/////////////////////////////////////

Ref<Resource> ResourceFormatLoaderShader::Load(const String &p_path,
                                               const String &p_original_path,
                                               CacheMode p_cache_mode) {
    String global_path = ProjectSettings::GetInstance()->GlobalizePath(ValidateLocalPath(p_path));
    QFile file(global_path);

    ERR_FAIL_COND_V_MSG(!file.exists(), {}, fmt::format("Cannot open file: {}.", p_path));
    ERR_FAIL_COND_V_MSG(!file.open(QIODevice::ReadOnly), {}, fmt::format("Cannot open file: {}.", p_path));
    auto byte_array = file.readAll();

    String extension = GetFileExtension(global_path);
    Ref<Shader> shader = MakeRef<Shader>();
    shader->SetShaderType(ExtensionsToShaderType(extension));
    shader->SetCode(file.readAll().data());
    return shader;
}

ShaderType ResourceFormatLoaderShader::ExtensionsToShaderType(const String& extension) {
    if(extension == ".vert") {
        return ShaderType::VertexShader;
    } else if (extension == ".frag") {
        return ShaderType::FragmentShader;
    } else if (extension == ".geom") {
        return ShaderType::GeometryShader;
    } else if (extension == "tesc") {
        return ShaderType::TessControlShader;
    } else if (extension == "tese") {
        return ShaderType::TessEvaluationShader;
    } else if (extension == ".comp") {
        return ShaderType::ComputeShader;
    } else {
        return ShaderType::None;
    }
}

void ResourceFormatLoaderShader::GetRecognizedExtensions(std::vector<String> *extensions) const {
    extensions->push_back(".vert");
    extensions->push_back(".frag");
    extensions->push_back(".geom");
    extensions->push_back(".tesc");
    extensions->push_back(".tese");
    extensions->push_back(".comp");
}

bool ResourceFormatLoaderShader::HandlesType(const String &type) const {
    return type == "Shader";
}


}