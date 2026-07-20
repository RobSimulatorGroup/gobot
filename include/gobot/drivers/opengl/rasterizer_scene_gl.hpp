/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-6-23
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "glad/glad.h"
#include "gobot/rendering/renderer_scene_render.hpp"
#include "gobot/rendering/luisa_renderer_module_api.hpp"
#include "gobot/rendering/scene_render_items.hpp"

#include <cstdint>
#include <array>
#include <memory>
#include <unordered_map>

namespace gobot::opengl {

struct RenderTarget;
struct LuisaRendererLifetime;

class GLRasterizerScene : public RendererSceneRender {
public:
    GLRasterizerScene();

    ~GLRasterizerScene() override;

    void RenderScene(const RID& render_target,
                     const RenderSceneSnapshot& scene,
                     const RenderViewSnapshot& view) override;

    [[nodiscard]] SceneRendererCapabilities GetCapabilities() const override;

    [[nodiscard]] SceneRendererStats GetStats() const override;

    void SetSettings(const SceneRendererSettings& settings) override;

    bool CaptureCudaRenderProduct(const RenderSceneSnapshot& scene,
                                  const RenderViewSnapshot& view,
                                  int width,
                                  int height,
                                  std::uint32_t output_mask,
                                  std::uint32_t mode,
                                  RendererRenderProductFrame* frame,
                                  std::string* error) override;

private:
    struct DefaultProgramUniforms {
        GLint model = -1;
        GLint normal_matrix = -1;
        GLint view_projection = -1;
        GLint view = -1;
        GLint color = -1;
        GLint camera_position = -1;
        GLint instance_id = -1;
        GLint semantic_id = -1;
        GLint metallic = -1;
        GLint roughness = -1;
        GLint specular = -1;
        GLint light_direction = -1;
        GLint light_color = -1;
        GLint light_intensity = -1;
        GLint sky_color = -1;
        GLint ground_color = -1;
        GLint ambient_intensity = -1;
        GLint exposure = -1;
        GLint normal_scale = -1;
        GLint occlusion_strength = -1;
        GLint emissive = -1;
        GLint alpha_mode = -1;
        GLint alpha_cutoff = -1;
        GLint has_albedo_texture = -1;
        GLint has_metallic_roughness_texture = -1;
        GLint has_normal_texture = -1;
        GLint has_occlusion_texture = -1;
        GLint has_emissive_texture = -1;
        GLint has_environment_texture = -1;
        GLint environment_rotation = -1;
        GLint environment_intensity = -1;
        GLint light_count = -1;
        std::array<GLint, 16> light_position_type{};
        std::array<GLint, 16> light_direction_range{};
        std::array<GLint, 16> light_color_intensity{};
        std::array<GLint, 16> light_spot_cosines{};
    };

    struct MeshCacheKey {
        std::uint64_t mesh_id = 0;
        std::uint64_t revision = 0;
        std::size_t surface_index = 0;

        bool operator==(const MeshCacheKey&) const = default;
    };

    struct MeshCacheKeyHash {
        std::size_t operator()(const MeshCacheKey& key) const;
    };

    struct MeshCacheEntry {
        GLuint vao = 0;
        GLuint vertex_buffer = 0;
        GLuint normal_buffer = 0;
        GLuint color_buffer = 0;
        GLuint tangent_buffer = 0;
        GLuint uv_buffer = 0;
        GLuint index_buffer = 0;
        GLsizei index_count = 0;
        std::uint64_t last_used_frame = 0;
    };

    struct TextureCacheKey {
        std::uint64_t texture_id = 0;
        std::uint64_t texture_revision = 0;
        std::uint64_t image_id = 0;
        std::uint64_t image_revision = 0;

        bool operator==(const TextureCacheKey&) const = default;
    };

    struct TextureCacheKeyHash {
        std::size_t operator()(const TextureCacheKey& key) const;
    };

    struct TextureCacheEntry {
        GLuint texture = 0;
        std::uint64_t last_used_frame = 0;
    };

    GLuint default_program_ = 0;
    DefaultProgramUniforms default_uniforms_;
    std::unordered_map<MeshCacheKey, MeshCacheEntry, MeshCacheKeyHash> mesh_cache_;
    std::unordered_map<TextureCacheKey, TextureCacheEntry, TextureCacheKeyHash> texture_cache_;
    std::uint64_t frame_index_ = 0;
    SceneRendererStats stats_;
    void* luisa_module_library_ = nullptr;
    void* luisa_renderer_ = nullptr;
    const LuisaRendererModuleApi* luisa_api_ = nullptr;
    std::shared_ptr<LuisaRendererLifetime> luisa_lifetime_;
    bool luisa_load_attempted_ = false;
    bool luisa_create_attempted_ = false;
    std::string luisa_status_;

    void EnsureDefaultProgram();

    void UploadFrameUniforms(const RenderSceneSnapshot& scene, const RenderViewSnapshot& view);

    MeshCacheEntry* GetOrCreateMesh(const VisualMeshRenderItem& item);

    GLuint GetOrCreateTexture(const RenderTextureSnapshot& texture);

    void DrawVisualItem(const VisualMeshRenderItem& item);

    bool TryLoadLuisaModule();

    bool EnsureLuisaRenderer();

    void UnloadLuisaModule();

    bool RenderWithLuisa(const RenderTarget& target,
                         const RenderSceneSnapshot& scene,
                         const RenderViewSnapshot& view);

    void RenderDepthPrepass(const RenderTarget& target,
                            const RenderSceneSnapshot& scene,
                            const RenderViewSnapshot& view);

    static void DestroyMeshEntry(MeshCacheEntry& entry);

    void PruneCaches();
};

}
