/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/scene/resources/shader.hpp"
#include "gobot/core/color.hpp"

namespace gobot {

// Material is a base Resource used for coloring and shading geometry.
class GOBOT_EXPORT Material : public Resource {
    GOBCLASS(Material, Resource)
public:
    Material();

    ~Material() override;

    RID GetRid() const override;

    virtual RID GetShaderRid() const;

    FORCE_INLINE RID GetMaterial() const { return material_; }

private:
    RID material_;
};

///////////////////////////////////////////////

class ShaderMaterial : public Material {
    GOBCLASS(ShaderMaterial, Material);
public:
    ShaderMaterial();

    ~ShaderMaterial();

    void SetShaderProgram(const Ref<ShaderProgram> &p_shader);

    Ref<ShaderProgram> GetShaderProgram() const;

    virtual RID GetShaderRid() const override;

private:
    Ref<ShaderProgram> shader_program_;
};

/////////////////////////////////////////////////////////////////


class GOBOT_EXPORT PBRMaterial3D: public Material {
    GOBCLASS(PBRMaterial3D, Material)
public:
    PBRMaterial3D();

    void SetAlbedo(const Color &albedo);

    Color GetAlbedo() const;

private:
    Color albedo_;

    Ref<RasterizerShaderProgram> shader_program_;
};

}
