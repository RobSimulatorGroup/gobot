/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-12
*/

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/io/resource_loader.hpp"

namespace gobot {

enum class ShaderType {
    None,
    VertexShader,
    FragmentShader,
    GeometryShader,
    ComputeShader,
    TessControlShader,
    TessEvaluationShader,
    Max
};

class GOBOT_EXPORT Shader : public Resource {
    GOBCLASS(Shader, Resource)
public:

    Shader();

    ~Shader() override;

    void SetCode(const String &p_code);

    String GetCode() const;

    void SetShaderType(ShaderType p_shader_type);

    ShaderType GetShaderType() const;

    RID GetRid() const override;

private:
    RID shader_;
    ShaderType shader_type_;
    String code_;
};

class GOBOT_EXPORT ShaderProgram : public Resource {
    GOBCLASS(ShaderProgram, Resource)
public:
    ShaderProgram();

    ~ShaderProgram() override;

    void SetAttachShaders(const std::vector<RID>& p_shaders);

    RID GetRid() const override;

private:
    RID shader_program_;
    std::vector<RID> shaders_;
};

class GOBOT_EXPORT ResourceFormatLoaderShader : public ResourceFormatLoader {
    GOBCLASS(ResourceFormatLoaderShader, ResourceFormatLoader)
public:
    Ref<Resource> Load(const String &p_path, const String &p_original_path = "", CacheMode p_cache_mode = CacheMode::Reuse) override;

    static ShaderType ExtensionsToShaderType(const String& extension);

    void GetRecognizedExtensions(std::vector<String> *extensions) const override;

    [[nodiscard]] bool HandlesType(const String &type) const override;
};

}