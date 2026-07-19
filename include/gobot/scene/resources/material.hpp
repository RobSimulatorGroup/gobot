/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/scene/resources/shader.hpp"
#include "gobot/scene/resources/texture.hpp"
#include "gobot/core/color.hpp"
#include "gobot/core/math/math_defs.hpp"

namespace gobot {

enum class AlphaMode {
    Opaque,
    Mask,
    Blend
};

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

    void SetMetallic(RealType metallic);

    RealType GetMetallic() const;

    void SetRoughness(RealType roughness);

    RealType GetRoughness() const;

    void SetSpecular(RealType specular);

    RealType GetSpecular() const;

    void SetAlbedoTexture(const Ref<Texture2D>& texture);

    const Ref<Texture2D>& GetAlbedoTexture() const;

    void SetMetallicRoughnessTexture(const Ref<Texture2D>& texture);

    const Ref<Texture2D>& GetMetallicRoughnessTexture() const;

    void SetNormalTexture(const Ref<Texture2D>& texture);

    const Ref<Texture2D>& GetNormalTexture() const;

    void SetNormalScale(RealType scale);

    RealType GetNormalScale() const;

    void SetOcclusionTexture(const Ref<Texture2D>& texture);

    const Ref<Texture2D>& GetOcclusionTexture() const;

    void SetOcclusionStrength(RealType strength);

    RealType GetOcclusionStrength() const;

    void SetEmissive(const Color& emissive);

    Color GetEmissive() const;

    void SetEmissiveTexture(const Ref<Texture2D>& texture);

    const Ref<Texture2D>& GetEmissiveTexture() const;

    void SetAlphaMode(AlphaMode mode);

    AlphaMode GetAlphaMode() const;

    void SetAlphaCutoff(RealType cutoff);

    RealType GetAlphaCutoff() const;

    void SetDoubleSided(bool double_sided);

    bool IsDoubleSided() const;

private:
    Color albedo_{0.8f, 0.8f, 0.8f, 1.0f};
    RealType metallic_ = 0.0f;
    RealType roughness_ = 0.5f;
    RealType specular_ = 0.5f;
    Ref<Texture2D> albedo_texture_;
    Ref<Texture2D> metallic_roughness_texture_;
    Ref<Texture2D> normal_texture_;
    RealType normal_scale_ = 1.0f;
    Ref<Texture2D> occlusion_texture_;
    RealType occlusion_strength_ = 1.0f;
    Color emissive_{0.0f, 0.0f, 0.0f, 1.0f};
    Ref<Texture2D> emissive_texture_;
    AlphaMode alpha_mode_ = AlphaMode::Opaque;
    RealType alpha_cutoff_ = 0.5f;
    bool double_sided_ = false;

    Ref<RasterizerShaderProgram> shader_program_;
};

}
