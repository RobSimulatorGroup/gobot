/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/


#include "gobot/scene/resources/material.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

Material::Material() {
    material_ = RS::GetInstance()->MaterialCreate();
}

Material::~Material() {
    RS::GetInstance()->Free(material_);
}

RID Material::GetRid() const {
    return material_;
}

RID Material::GetShaderRid() const {
    return {};
}

/////////////////////

ShaderMaterial::ShaderMaterial() {

}

ShaderMaterial::~ShaderMaterial() {

}

void ShaderMaterial::SetShader(const Ref<Shader> &p_shader) {
    shader_ = p_shader;
}

Ref<Shader> ShaderMaterial::GetShader() const {
    return shader_;
}

RID ShaderMaterial::GetShaderRid() const {
    if (shader_.IsValid()) {
        return shader_->GetRid();
    } else {
        return RID();
    }
}



///////////////////////////////

PBRMaterial3D::PBRMaterial3D() {

}

void PBRMaterial3D::SetAlbedo(const Color &albedo) {
    albedo_ = albedo;
}

Color PBRMaterial3D::GetAlbedo() const {
    return albedo_;
}

}

GOBOT_REGISTRATION {
    Class_<Material>("Material")
        .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<Material>, Ref<Resource>>();

    Class_<PBRMaterial3D>("PBRMaterial3D")
        .constructor()(CtorAsRawPtr)
        .property("albedo", &PBRMaterial3D::GetAlbedo, &PBRMaterial3D::SetAlbedo);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PBRMaterial3D>, Ref<Material>>();

};
