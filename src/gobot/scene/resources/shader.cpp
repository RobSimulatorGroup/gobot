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

namespace gobot {

Shader::Shader() {
    shader_ = RS::GetInstance()->ShaderCreate();
}

Shader::~Shader() {
    RS::GetInstance()->ShaderFree(shader_);
}

void Shader::SetCode(const String &p_code) {
    RS::GetInstance()->ShaderSetCode(shader_, p_code);
}

String Shader::GetCode() const {
    return RS::GetInstance()->ShaderGetCode(shader_);
}

ShaderType Shader::GetShaderType() const {
    return shader_type_;
}

RID Shader::GetRid() const {
    return shader_;
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
    shader->SetCode(file.readAll().data());
    return shader;
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