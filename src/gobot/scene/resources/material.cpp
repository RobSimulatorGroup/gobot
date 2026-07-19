/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
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
    if (albedo_ == albedo) {
        return;
    }
    albedo_ = albedo;
    MarkChanged();
}

Color PBRMaterial3D::GetAlbedo() const {
    return albedo_;
}

void PBRMaterial3D::SetMetallic(RealType metallic) {
    const RealType clamped = std::clamp(metallic, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
    if (metallic_ != clamped) {
        metallic_ = clamped;
        MarkChanged();
    }
}

RealType PBRMaterial3D::GetMetallic() const {
    return metallic_;
}

void PBRMaterial3D::SetRoughness(RealType roughness) {
    const RealType clamped = std::clamp(roughness, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
    if (roughness_ != clamped) {
        roughness_ = clamped;
        MarkChanged();
    }
}

RealType PBRMaterial3D::GetRoughness() const {
    return roughness_;
}

void PBRMaterial3D::SetSpecular(RealType specular) {
    const RealType clamped = std::clamp(specular, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
    if (specular_ != clamped) {
        specular_ = clamped;
        MarkChanged();
    }
}

RealType PBRMaterial3D::GetSpecular() const {
    return specular_;
}

void PBRMaterial3D::SetAlbedoTexture(const Ref<Texture2D>& texture) {
    if (albedo_texture_.Get() != texture.Get()) {
        albedo_texture_ = texture;
        MarkChanged();
    }
}

const Ref<Texture2D>& PBRMaterial3D::GetAlbedoTexture() const { return albedo_texture_; }

void PBRMaterial3D::SetMetallicRoughnessTexture(const Ref<Texture2D>& texture) {
    if (metallic_roughness_texture_.Get() != texture.Get()) {
        metallic_roughness_texture_ = texture;
        MarkChanged();
    }
}

const Ref<Texture2D>& PBRMaterial3D::GetMetallicRoughnessTexture() const { return metallic_roughness_texture_; }

void PBRMaterial3D::SetNormalTexture(const Ref<Texture2D>& texture) {
    if (normal_texture_.Get() != texture.Get()) {
        normal_texture_ = texture;
        MarkChanged();
    }
}

const Ref<Texture2D>& PBRMaterial3D::GetNormalTexture() const { return normal_texture_; }

void PBRMaterial3D::SetNormalScale(RealType scale) {
    const RealType clamped = std::max<RealType>(0.0, scale);
    if (normal_scale_ != clamped) {
        normal_scale_ = clamped;
        MarkChanged();
    }
}

RealType PBRMaterial3D::GetNormalScale() const { return normal_scale_; }

void PBRMaterial3D::SetOcclusionTexture(const Ref<Texture2D>& texture) {
    if (occlusion_texture_.Get() != texture.Get()) {
        occlusion_texture_ = texture;
        MarkChanged();
    }
}

const Ref<Texture2D>& PBRMaterial3D::GetOcclusionTexture() const { return occlusion_texture_; }

void PBRMaterial3D::SetOcclusionStrength(RealType strength) {
    const RealType clamped = std::clamp(strength, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
    if (occlusion_strength_ != clamped) {
        occlusion_strength_ = clamped;
        MarkChanged();
    }
}

RealType PBRMaterial3D::GetOcclusionStrength() const { return occlusion_strength_; }

void PBRMaterial3D::SetEmissive(const Color& emissive) {
    if (emissive_ != emissive) {
        emissive_ = emissive;
        MarkChanged();
    }
}

Color PBRMaterial3D::GetEmissive() const { return emissive_; }

void PBRMaterial3D::SetEmissiveTexture(const Ref<Texture2D>& texture) {
    if (emissive_texture_.Get() != texture.Get()) {
        emissive_texture_ = texture;
        MarkChanged();
    }
}

const Ref<Texture2D>& PBRMaterial3D::GetEmissiveTexture() const { return emissive_texture_; }

void PBRMaterial3D::SetAlphaMode(AlphaMode mode) {
    if (alpha_mode_ != mode) {
        alpha_mode_ = mode;
        MarkChanged();
    }
}

AlphaMode PBRMaterial3D::GetAlphaMode() const { return alpha_mode_; }

void PBRMaterial3D::SetAlphaCutoff(RealType cutoff) {
    const RealType clamped = std::clamp(cutoff, static_cast<RealType>(0.0), static_cast<RealType>(1.0));
    if (alpha_cutoff_ != clamped) {
        alpha_cutoff_ = clamped;
        MarkChanged();
    }
}

RealType PBRMaterial3D::GetAlphaCutoff() const { return alpha_cutoff_; }

void PBRMaterial3D::SetDoubleSided(bool double_sided) {
    if (double_sided_ != double_sided) {
        double_sided_ = double_sided;
        MarkChanged();
    }
}

bool PBRMaterial3D::IsDoubleSided() const { return double_sided_; }

}

GOBOT_REGISTRATION {
    QuickEnumeration_<AlphaMode>("AlphaMode");

    Class_<Material>("Material")
        .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<Material>, Ref<Resource>>();

    Class_<PBRMaterial3D>("PBRMaterial3D")
        .constructor()(CtorAsRawPtr)
        .property("albedo", &PBRMaterial3D::GetAlbedo, &PBRMaterial3D::SetAlbedo)
        .property("metallic", &PBRMaterial3D::GetMetallic, &PBRMaterial3D::SetMetallic)
        .property("roughness", &PBRMaterial3D::GetRoughness, &PBRMaterial3D::SetRoughness)
        .property("specular", &PBRMaterial3D::GetSpecular, &PBRMaterial3D::SetSpecular)
        .property("albedo_texture", &PBRMaterial3D::GetAlbedoTexture, &PBRMaterial3D::SetAlbedoTexture)
        .property("metallic_roughness_texture", &PBRMaterial3D::GetMetallicRoughnessTexture, &PBRMaterial3D::SetMetallicRoughnessTexture)
        .property("normal_texture", &PBRMaterial3D::GetNormalTexture, &PBRMaterial3D::SetNormalTexture)
        .property("normal_scale", &PBRMaterial3D::GetNormalScale, &PBRMaterial3D::SetNormalScale)
        .property("occlusion_texture", &PBRMaterial3D::GetOcclusionTexture, &PBRMaterial3D::SetOcclusionTexture)
        .property("occlusion_strength", &PBRMaterial3D::GetOcclusionStrength, &PBRMaterial3D::SetOcclusionStrength)
        .property("emissive", &PBRMaterial3D::GetEmissive, &PBRMaterial3D::SetEmissive)
        .property("emissive_texture", &PBRMaterial3D::GetEmissiveTexture, &PBRMaterial3D::SetEmissiveTexture)
        .property("alpha_mode", &PBRMaterial3D::GetAlphaMode, &PBRMaterial3D::SetAlphaMode)
        .property("alpha_cutoff", &PBRMaterial3D::GetAlphaCutoff, &PBRMaterial3D::SetAlphaCutoff)
        .property("double_sided", &PBRMaterial3D::IsDoubleSided, &PBRMaterial3D::SetDoubleSided);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PBRMaterial3D>, Ref<Material>>();

};
