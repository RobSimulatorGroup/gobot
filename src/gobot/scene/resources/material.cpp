/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */


#include "gobot/scene/resources/material.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/rendering/render_server.hpp"

#include <algorithm>

namespace gobot {

Material::Material() {
    if (RenderServer::HasInstance()) {
        material_ = RS::GetInstance()->MaterialCreate();
    }
}

Material::~Material() {
    if (RenderServer::HasInstance() && material_.IsValid()) {
        RS::GetInstance()->Free(material_);
    }
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

void ShaderMaterial::SetShaderProgram(const Ref<ShaderProgram> &p_shader_program_) {
    shader_program_ = p_shader_program_;
}

Ref<ShaderProgram> ShaderMaterial::GetShaderProgram() const {
    return shader_program_;
}

RID ShaderMaterial::GetShaderRid() const {
    if (shader_program_.IsValid()) {
        return shader_program_->GetRid();
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

void PBRMaterial3D::SetMetallic(RealType metallic) {
    metallic_ = std::clamp(metallic, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
}

RealType PBRMaterial3D::GetMetallic() const {
    return metallic_;
}

void PBRMaterial3D::SetRoughness(RealType roughness) {
    roughness_ = std::clamp(roughness, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
}

RealType PBRMaterial3D::GetRoughness() const {
    return roughness_;
}

void PBRMaterial3D::SetSpecular(RealType specular) {
    specular_ = std::clamp(specular, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
}

RealType PBRMaterial3D::GetSpecular() const {
    return specular_;
}

}

GOBOT_REGISTRATION {
    Class_<Material>("Material")
        .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<Material>, Ref<Resource>>();

    Class_<PBRMaterial3D>("PBRMaterial3D")
        .constructor()(CtorAsRawPtr)
        .property("albedo", &PBRMaterial3D::GetAlbedo, &PBRMaterial3D::SetAlbedo)
        .property("metallic", &PBRMaterial3D::GetMetallic, &PBRMaterial3D::SetMetallic)
        .property("roughness", &PBRMaterial3D::GetRoughness, &PBRMaterial3D::SetRoughness)
        .property("specular", &PBRMaterial3D::GetSpecular, &PBRMaterial3D::SetSpecular);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PBRMaterial3D>, Ref<Material>>();

};
