/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/drivers/opengl/rasterizer_scene_gl.hpp"

#include "gobot/core/profile.hpp"
#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"
#include "glsl_shader_hpp/default_mesh_frag.hpp"
#include "glsl_shader_hpp/default_mesh_vert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace gobot::opengl {

struct LuisaRendererLifetime {
    void* library = nullptr;
    void* renderer = nullptr;
    const LuisaRendererModuleApi* api = nullptr;

    ~LuisaRendererLifetime() {
        if (renderer != nullptr && api != nullptr && api->destroy != nullptr) {
            api->destroy(renderer);
        }
        if (library != nullptr) {
            dlclose(library);
        }
    }
};

namespace {

void LuisaModuleAnchor() {}

void AddModuleCandidateDirectory(std::vector<std::filesystem::path>* candidates,
                                 const std::filesystem::path& directory) {
    if (!directory.empty()) {
        candidates->emplace_back(directory / "libgobot_luisa_renderer.so");
    }
}

GLuint CompileShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLsizei log_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::string log(static_cast<std::size_t>(std::max(log_length, GLsizei{1})), '\0');
        glGetShaderInfoLog(shader, log_length, &log_length, log.data());
        LOG_ERROR("Default mesh shader compilation failed: {}", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

std::size_t CombineHash(std::size_t seed, std::uint64_t value) {
    return seed ^ (std::hash<std::uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

GLenum ToGLWrap(TextureWrap wrap) {
    switch (wrap) {
        case TextureWrap::ClampToEdge:
            return GL_CLAMP_TO_EDGE;
        case TextureWrap::MirroredRepeat:
            return GL_MIRRORED_REPEAT;
        case TextureWrap::Repeat:
        default:
            return GL_REPEAT;
    }
}

bool ResolveImageFormat(ImageFormat format, GLenum* internal_format, GLenum* pixel_format, GLenum* pixel_type) {
    switch (format) {
        case ImageFormat::L8:
        case ImageFormat::R8:
            *internal_format = GL_R8;
            *pixel_format = GL_RED;
            *pixel_type = GL_UNSIGNED_BYTE;
            return true;
        case ImageFormat::LA8:
        case ImageFormat::RG8:
            *internal_format = GL_RG8;
            *pixel_format = GL_RG;
            *pixel_type = GL_UNSIGNED_BYTE;
            return true;
        case ImageFormat::RGB8:
            *internal_format = GL_RGB8;
            *pixel_format = GL_RGB;
            *pixel_type = GL_UNSIGNED_BYTE;
            return true;
        case ImageFormat::RGBA8:
            *internal_format = GL_RGBA8;
            *pixel_format = GL_RGBA;
            *pixel_type = GL_UNSIGNED_BYTE;
            return true;
        case ImageFormat::RF:
            *internal_format = GL_R32F;
            *pixel_format = GL_RED;
            *pixel_type = GL_FLOAT;
            return true;
        case ImageFormat::RGF:
            *internal_format = GL_RG32F;
            *pixel_format = GL_RG;
            *pixel_type = GL_FLOAT;
            return true;
        case ImageFormat::RGBF:
            *internal_format = GL_RGB32F;
            *pixel_format = GL_RGB;
            *pixel_type = GL_FLOAT;
            return true;
        case ImageFormat::RGBAF:
            *internal_format = GL_RGBA32F;
            *pixel_format = GL_RGBA;
            *pixel_type = GL_FLOAT;
            return true;
        case ImageFormat::RH:
            *internal_format = GL_R16F;
            *pixel_format = GL_RED;
            *pixel_type = GL_HALF_FLOAT;
            return true;
        case ImageFormat::RGH:
            *internal_format = GL_RG16F;
            *pixel_format = GL_RG;
            *pixel_type = GL_HALF_FLOAT;
            return true;
        case ImageFormat::RGBH:
            *internal_format = GL_RGB16F;
            *pixel_format = GL_RGB;
            *pixel_type = GL_HALF_FLOAT;
            return true;
        case ImageFormat::RGBAH:
            *internal_format = GL_RGBA16F;
            *pixel_format = GL_RGBA;
            *pixel_type = GL_HALF_FLOAT;
            return true;
        default:
            return false;
    }
}

void PackVector3(const std::vector<Vector3>& source, std::vector<float>* destination) {
    destination->reserve(source.size() * 3);
    for (const Vector3& value : source) {
        destination->push_back(static_cast<float>(value.x()));
        destination->push_back(static_cast<float>(value.y()));
        destination->push_back(static_cast<float>(value.z()));
    }
}

} // namespace

std::size_t GLRasterizerScene::MeshCacheKeyHash::operator()(const MeshCacheKey& key) const {
    std::size_t hash = CombineHash(0, key.mesh_id);
    hash = CombineHash(hash, key.revision);
    return CombineHash(hash, key.surface_index);
}

std::size_t GLRasterizerScene::TextureCacheKeyHash::operator()(const TextureCacheKey& key) const {
    std::size_t hash = CombineHash(0, key.texture_id);
    hash = CombineHash(hash, key.texture_revision);
    hash = CombineHash(hash, key.image_id);
    return CombineHash(hash, key.image_revision);
}

void GLRasterizerScene::DestroyMeshEntry(MeshCacheEntry& entry) {
    if (entry.vao != 0) {
        glDeleteVertexArrays(1, &entry.vao);
    }
    const std::array<GLuint, 6> buffers = {
            entry.vertex_buffer,
            entry.normal_buffer,
            entry.color_buffer,
            entry.tangent_buffer,
            entry.uv_buffer,
            entry.index_buffer};
    for (const GLuint buffer : buffers) {
        if (buffer != 0) {
            glDeleteBuffers(1, &buffer);
        }
    }
    entry = {};
}

GLRasterizerScene::GLRasterizerScene() = default;

SceneRendererCapabilities GLRasterizerScene::GetCapabilities() const {
    auto* self = const_cast<GLRasterizerScene*>(this);
    if (self->TryLoadLuisaModule() && self->luisa_api_->capabilities != nullptr) {
        SceneRendererCapabilities capabilities = self->luisa_api_->capabilities(nullptr);
        capabilities.cuda_render_products = capabilities.cuda_render_products &&
                                            self->luisa_api_->capture_render_product != nullptr &&
                                            self->luisa_api_->release_render_product != nullptr &&
                                            self->luisa_api_->readback_render_product != nullptr;
        if (self->luisa_create_attempted_ && self->luisa_renderer_ == nullptr) {
            capabilities.ray_tracing_available = false;
            capabilities.realtime = false;
            capabilities.progressive = false;
            capabilities.cuda_render_products = false;
            capabilities.status = self->luisa_status_;
        }
        return capabilities;
    }
    SceneRendererCapabilities fallback;
    fallback.backend_name = "OpenGL 4.6";
    fallback.status = luisa_status_.empty()
                              ? "Optional LuisaCompute renderer module is not available."
                              : luisa_status_;
    return fallback;
}

SceneRendererStats GLRasterizerScene::GetStats() const {
    return stats_;
}

bool GLRasterizerScene::CaptureCudaRenderProduct(const RenderSceneSnapshot& scene,
                                                 const RenderViewSnapshot& view,
                                                 int width,
                                                 int height,
                                                 std::uint32_t output_mask,
                                                 std::uint32_t mode,
                                                 RendererRenderProductFrame* frame,
                                                 std::string* error) {
    if (frame == nullptr) {
        if (error != nullptr) {
            *error = "CUDA render-product destination is null";
        }
        return false;
    }
    if (!EnsureLuisaRenderer() || luisa_api_ == nullptr ||
        luisa_api_->capture_render_product == nullptr ||
        luisa_api_->release_render_product == nullptr ||
        luisa_api_->readback_render_product == nullptr) {
        if (error != nullptr) {
            *error = luisa_status_.empty()
                             ? "Luisa CUDA render-product API is unavailable"
                             : luisa_status_;
        }
        return false;
    }

    const LuisaRenderProductRequest request{width, height, output_mask, mode};
    LuisaRenderProductFrame module_frame;
    std::array<char, 1024> module_error{};
    const LuisaRendererResult result = luisa_api_->capture_render_product(
            luisa_renderer_,
            &scene,
            &view,
            &request,
            &module_frame,
            module_error.data(),
            module_error.size());
    if (result != LuisaRendererResult::Success || module_frame.frame == nullptr) {
        if (error != nullptr) {
            *error = module_error[0] != '\0'
                             ? module_error.data()
                             : "Luisa CUDA render-product capture failed";
        }
        if (result == LuisaRendererResult::FatalError) {
            luisa_status_ = error != nullptr ? *error : "Luisa CUDA render-product fatal error";
        }
        return false;
    }

    const LuisaRendererModuleApi* api = luisa_api_;
    void* renderer = luisa_renderer_;
    void* handle = module_frame.frame;
    const std::shared_ptr<LuisaRendererLifetime> lifetime = luisa_lifetime_;
    auto owner = std::shared_ptr<void>(handle, [api, renderer, lifetime](void* product_frame) {
        api->release_render_product(renderer, product_frame);
    });
    frame->owner = owner;
    frame->device_id = module_frame.device_id;
    for (std::size_t index = 0; index < frame->buffers.size(); ++index) {
        frame->buffers[index] = {
                module_frame.buffers[index].device_pointer,
                module_frame.buffers[index].allocation_size,
                module_frame.buffers[index].pixel_stride_bytes};
    }
    frame->copy_to_host = [api, renderer, handle, owner](std::uint32_t output,
                                                         void* destination,
                                                         std::size_t destination_size) {
        std::array<char, 1024> readback_error{};
        return api->readback_render_product(renderer,
                                            handle,
                                            output,
                                            destination,
                                            destination_size,
                                            readback_error.data(),
                                            readback_error.size());
    };
    return true;
}

void GLRasterizerScene::SetSettings(const SceneRendererSettings& settings) {
    const SceneRendererMode previous_mode = settings_.mode;
    settings_ = settings;
    settings_.target_fps = std::clamp(settings_.target_fps, 1, 240);
    settings_.samples_per_frame = std::clamp(settings_.samples_per_frame, 1, 1024);
    settings_.max_accumulated_samples = std::max(settings_.max_accumulated_samples, 1);
    settings_.max_bounces = std::clamp(settings_.max_bounces, 1, 32);
    if (settings_.mode != SceneRendererMode::Raster) {
        EnsureLuisaRenderer();
    }
    if (previous_mode != settings_.mode && luisa_api_ != nullptr &&
        luisa_api_->reset_accumulation != nullptr) {
        luisa_api_->reset_accumulation(luisa_renderer_);
    }
}

GLRasterizerScene::~GLRasterizerScene() {
    UnloadLuisaModule();
    for (auto& [key, entry] : mesh_cache_) {
        DestroyMeshEntry(entry);
    }
    for (auto& [key, entry] : texture_cache_) {
        if (entry.texture != 0) {
            glDeleteTextures(1, &entry.texture);
        }
    }
    if (default_program_ != 0) {
        glDeleteProgram(default_program_);
    }
}

void GLRasterizerScene::RenderScene(const RID& render_target,
                                    const RenderSceneSnapshot& scene,
                                    const RenderViewSnapshot& view) {
    GOBOT_PROFILE_ZONE("OpenGL::RenderScene");
    auto* rt = TextureStorage::GetInstance()->GetRenderTarget(render_target);
    ERR_FAIL_COND(rt == nullptr);
    if (rt->fbo == 0 || rt->size.x() <= 0 || rt->size.y() <= 0) {
        return;
    }

    EnsureDefaultProgram();
    if (default_program_ == 0) {
        return;
    }

    ++frame_index_;
    if (view.mode == RenderViewMode::Viewport &&
        settings_.mode != SceneRendererMode::Raster && EnsureLuisaRenderer()) {
        RenderDepthPrepass(*rt, scene, view);
        if (RenderWithLuisa(*rt, scene, view)) {
            PruneCaches();
            return;
        }
    }
    stats_.active_mode = SceneRendererMode::Raster;
    stats_.accumulated_samples = 0;
    stats_.status = settings_.mode == SceneRendererMode::Raster
                            ? "OpenGL raster"
                            : "OpenGL raster fallback: " +
                                      (luisa_status_.empty()
                                               ? std::string{"LuisaCompute renderer unavailable."}
                                               : luisa_status_);
    glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);
    glViewport(0, 0, rt->size.x(), rt->size.y());
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glFrontFace(GL_CCW);
    const Color& clear_color = scene.environment.clear_color;
    glClear(GL_DEPTH_BUFFER_BIT);
    if (RenderOutputMaskContains(rt->output_mask, RenderOutputType::Rgb)) {
        const std::array<float, 4> value = {
                clear_color.red(), clear_color.green(), clear_color.blue(), clear_color.alpha()};
        glClearBufferfv(GL_COLOR, static_cast<GLint>(RenderOutputType::Rgb), value.data());
    }
    if (RenderOutputMaskContains(rt->output_mask, RenderOutputType::LinearDepth)) {
        const std::array<float, 4> value = {
                std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f};
        glClearBufferfv(GL_COLOR, static_cast<GLint>(RenderOutputType::LinearDepth), value.data());
    }
    if (RenderOutputMaskContains(rt->output_mask, RenderOutputType::WorldNormal)) {
        const std::array<float, 4> value{};
        glClearBufferfv(GL_COLOR, static_cast<GLint>(RenderOutputType::WorldNormal), value.data());
    }
    const std::array<std::uint32_t, 4> zero_ids{};
    if (RenderOutputMaskContains(rt->output_mask, RenderOutputType::InstanceId)) {
        glClearBufferuiv(GL_COLOR, static_cast<GLint>(RenderOutputType::InstanceId), zero_ids.data());
    }
    if (RenderOutputMaskContains(rt->output_mask, RenderOutputType::SemanticId)) {
        glClearBufferuiv(GL_COLOR, static_cast<GLint>(RenderOutputType::SemanticId), zero_ids.data());
    }

    glUseProgram(default_program_);
    UploadFrameUniforms(scene, view);
    for (const VisualMeshRenderItem& item : scene.visual_meshes) {
        if (item.material.alpha_mode != AlphaMode::Blend) {
            DrawVisualItem(item);
        }
    }
    for (const VisualMeshRenderItem& item : scene.visual_meshes) {
        if (item.material.alpha_mode == AlphaMode::Blend) {
            DrawVisualItem(item);
        }
    }

    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    PruneCaches();
}

bool GLRasterizerScene::TryLoadLuisaModule() {
    if (luisa_api_ != nullptr) {
        return true;
    }
    if (luisa_load_attempted_) {
        return false;
    }
    luisa_load_attempted_ = true;
    luisa_lifetime_ = std::make_shared<LuisaRendererLifetime>();

    std::vector<std::filesystem::path> candidates;
    if (const char* explicit_path = std::getenv("GOBOT_LUISA_RENDERER_LIBRARY");
        explicit_path != nullptr && explicit_path[0] != '\0') {
        candidates.emplace_back(explicit_path);
    }
    candidates.emplace_back("libgobot_luisa_renderer.so");
    candidates.emplace_back(std::filesystem::current_path() / "libgobot_luisa_renderer.so");
    candidates.emplace_back(std::filesystem::current_path() / "build" / "libgobot_luisa_renderer.so");
    Dl_info module_info{};
    if (dladdr(reinterpret_cast<void*>(&LuisaModuleAnchor), &module_info) != 0 &&
        module_info.dli_fname != nullptr) {
        AddModuleCandidateDirectory(&candidates,
                                    std::filesystem::path(module_info.dli_fname).parent_path());
    }
    std::error_code executable_error;
    const std::filesystem::path executable = std::filesystem::read_symlink(
            "/proc/self/exe", executable_error);
    if (!executable_error) {
        AddModuleCandidateDirectory(&candidates, executable.parent_path());
    }

    for (const std::filesystem::path& candidate : candidates) {
        luisa_module_library_ = dlopen(candidate.string().c_str(), RTLD_NOW | RTLD_LOCAL);
        if (luisa_module_library_ != nullptr) {
            luisa_lifetime_->library = luisa_module_library_;
            break;
        }
    }
    if (luisa_module_library_ == nullptr) {
        luisa_status_ = "LuisaCompute module not found; configure GOB_BUILD_LUISA_RENDERER=ON or set GOBOT_LUISA_RENDERER_LIBRARY.";
        return false;
    }

    auto get_api = reinterpret_cast<GetLuisaRendererModuleApi>(
            dlsym(luisa_module_library_, "gobot_luisa_renderer_get_api"));
    if (get_api == nullptr) {
        luisa_status_ = "LuisaCompute module is missing gobot_luisa_renderer_get_api.";
        UnloadLuisaModule();
        luisa_load_attempted_ = true;
        return false;
    }
    luisa_api_ = get_api();
    if (luisa_api_ == nullptr || luisa_api_->abi_version != GOBOT_LUISA_RENDERER_ABI_VERSION ||
        luisa_api_->create == nullptr || luisa_api_->destroy == nullptr || luisa_api_->render == nullptr) {
        luisa_status_ = "LuisaCompute module ABI does not match this Gobot build.";
        UnloadLuisaModule();
        luisa_load_attempted_ = true;
        return false;
    }
    luisa_lifetime_->api = luisa_api_;

    luisa_status_ = "LuisaCompute CUDA renderer module available.";
    return true;
}

bool GLRasterizerScene::EnsureLuisaRenderer() {
    if (luisa_renderer_ != nullptr) {
        return true;
    }
    if (luisa_create_attempted_) {
        return false;
    }
    if (!TryLoadLuisaModule()) {
        return false;
    }
    luisa_create_attempted_ = true;
    Dl_info module_info{};
    std::filesystem::path module_directory = std::filesystem::current_path();
    if (dladdr(reinterpret_cast<void*>(luisa_api_->render), &module_info) != 0 &&
        module_info.dli_fname != nullptr) {
        module_directory = std::filesystem::path(module_info.dli_fname).parent_path();
    }
    std::array<char, 1024> error{};
    luisa_renderer_ = luisa_api_->create(
            module_directory.string().c_str(), error.data(), error.size());
    if (luisa_renderer_ == nullptr) {
        luisa_status_ = error[0] != '\0'
                                ? error.data()
                                : "LuisaCompute CUDA device initialization failed.";
        return false;
    }
    luisa_lifetime_->renderer = luisa_renderer_;
    luisa_status_ = "LuisaCompute CUDA renderer loaded.";
    return true;
}

void GLRasterizerScene::UnloadLuisaModule() {
    luisa_renderer_ = nullptr;
    luisa_api_ = nullptr;
    luisa_module_library_ = nullptr;
    luisa_lifetime_.reset();
}

bool GLRasterizerScene::RenderWithLuisa(const RenderTarget& target,
                                        const RenderSceneSnapshot& scene,
                                        const RenderViewSnapshot& view) {
    if (luisa_api_ == nullptr || luisa_renderer_ == nullptr) {
        return false;
    }
    const LuisaRendererTarget render_target{
            target.color,
            target.depth,
            target.size.x(),
            target.size.y()};
    std::array<char, 1024> error{};
    const LuisaRendererResult result = luisa_api_->render(
            luisa_renderer_,
            &render_target,
            &scene,
            &view,
            &settings_,
            &stats_,
            error.data(),
            error.size());
    if (result == LuisaRendererResult::Success) {
        return true;
    }
    luisa_status_ = error[0] != '\0' ? error.data() : "LuisaCompute renderer failed; using raster fallback.";
    stats_.status = luisa_status_;
    if (result == LuisaRendererResult::FatalError) {
        UnloadLuisaModule();
        luisa_load_attempted_ = true;
    }
    return false;
}

void GLRasterizerScene::RenderDepthPrepass(const RenderTarget& target,
                                           const RenderSceneSnapshot& scene,
                                           const RenderViewSnapshot& view) {
    glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
    glViewport(0, 0, target.size.x(), target.size.y());
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glUseProgram(default_program_);
    UploadFrameUniforms(scene, view);
    for (const VisualMeshRenderItem& item : scene.visual_meshes) {
        if (item.material.alpha_mode != AlphaMode::Blend) {
            DrawVisualItem(item);
        }
    }
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLRasterizerScene::EnsureDefaultProgram() {
    if (default_program_ != 0) {
        return;
    }

    const GLuint vs = CompileShader(GL_VERTEX_SHADER, DEFAULT_MESH_VERT);
    const GLuint fs = CompileShader(GL_FRAGMENT_SHADER, DEFAULT_MESH_FRAG);
    if (vs == 0 || fs == 0) {
        if (vs != 0) {
            glDeleteShader(vs);
        }
        if (fs != 0) {
            glDeleteShader(fs);
        }
        return;
    }

    default_program_ = glCreateProgram();
    glAttachShader(default_program_, vs);
    glAttachShader(default_program_, fs);
    glLinkProgram(default_program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint status = GL_FALSE;
    glGetProgramiv(default_program_, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLsizei log_length = 0;
        glGetProgramiv(default_program_, GL_INFO_LOG_LENGTH, &log_length);
        std::string log(static_cast<std::size_t>(std::max(log_length, GLsizei{1})), '\0');
        glGetProgramInfoLog(default_program_, log_length, &log_length, log.data());
        LOG_ERROR("Default mesh program link failed: {}", log);
        glDeleteProgram(default_program_);
        default_program_ = 0;
        return;
    }

    default_uniforms_.model = glGetUniformLocation(default_program_, "u_model");
    default_uniforms_.normal_matrix = glGetUniformLocation(default_program_, "u_normal_matrix");
    default_uniforms_.view_projection = glGetUniformLocation(default_program_, "u_view_projection");
    default_uniforms_.view = glGetUniformLocation(default_program_, "u_view");
    default_uniforms_.color = glGetUniformLocation(default_program_, "u_color");
    default_uniforms_.camera_position = glGetUniformLocation(default_program_, "u_camera_position");
    default_uniforms_.instance_id = glGetUniformLocation(default_program_, "u_instance_id");
    default_uniforms_.semantic_id = glGetUniformLocation(default_program_, "u_semantic_id");
    default_uniforms_.metallic = glGetUniformLocation(default_program_, "u_metallic");
    default_uniforms_.roughness = glGetUniformLocation(default_program_, "u_roughness");
    default_uniforms_.specular = glGetUniformLocation(default_program_, "u_specular");
    default_uniforms_.light_direction = glGetUniformLocation(default_program_, "u_light_direction");
    default_uniforms_.light_color = glGetUniformLocation(default_program_, "u_light_color");
    default_uniforms_.light_intensity = glGetUniformLocation(default_program_, "u_light_intensity");
    default_uniforms_.sky_color = glGetUniformLocation(default_program_, "u_sky_color");
    default_uniforms_.ground_color = glGetUniformLocation(default_program_, "u_ground_color");
    default_uniforms_.ambient_intensity = glGetUniformLocation(default_program_, "u_ambient_intensity");
    default_uniforms_.exposure = glGetUniformLocation(default_program_, "u_exposure");
    default_uniforms_.normal_scale = glGetUniformLocation(default_program_, "u_normal_scale");
    default_uniforms_.occlusion_strength = glGetUniformLocation(default_program_, "u_occlusion_strength");
    default_uniforms_.emissive = glGetUniformLocation(default_program_, "u_emissive");
    default_uniforms_.alpha_mode = glGetUniformLocation(default_program_, "u_alpha_mode");
    default_uniforms_.alpha_cutoff = glGetUniformLocation(default_program_, "u_alpha_cutoff");
    default_uniforms_.has_albedo_texture = glGetUniformLocation(default_program_, "u_has_albedo_texture");
    default_uniforms_.has_metallic_roughness_texture =
            glGetUniformLocation(default_program_, "u_has_metallic_roughness_texture");
    default_uniforms_.has_normal_texture = glGetUniformLocation(default_program_, "u_has_normal_texture");
    default_uniforms_.has_occlusion_texture = glGetUniformLocation(default_program_, "u_has_occlusion_texture");
    default_uniforms_.has_emissive_texture = glGetUniformLocation(default_program_, "u_has_emissive_texture");
    default_uniforms_.has_environment_texture =
            glGetUniformLocation(default_program_, "u_has_environment_texture");
    default_uniforms_.environment_rotation =
            glGetUniformLocation(default_program_, "u_environment_rotation");
    default_uniforms_.environment_intensity =
            glGetUniformLocation(default_program_, "u_environment_intensity");
    default_uniforms_.light_count = glGetUniformLocation(default_program_, "u_light_count");
    for (std::size_t i = 0; i < default_uniforms_.light_position_type.size(); ++i) {
        const std::string index = std::to_string(i);
        default_uniforms_.light_position_type[i] =
                glGetUniformLocation(default_program_, ("u_light_position_type[" + index + "]").c_str());
        default_uniforms_.light_direction_range[i] =
                glGetUniformLocation(default_program_, ("u_light_direction_range[" + index + "]").c_str());
        default_uniforms_.light_color_intensity[i] =
                glGetUniformLocation(default_program_, ("u_light_color_intensity[" + index + "]").c_str());
        default_uniforms_.light_spot_cosines[i] =
                glGetUniformLocation(default_program_, ("u_light_spot_cosines[" + index + "]").c_str());
    }

    glUseProgram(default_program_);
    glUniform1i(glGetUniformLocation(default_program_, "u_albedo_texture"), 0);
    glUniform1i(glGetUniformLocation(default_program_, "u_metallic_roughness_texture"), 1);
    glUniform1i(glGetUniformLocation(default_program_, "u_normal_texture"), 2);
    glUniform1i(glGetUniformLocation(default_program_, "u_occlusion_texture"), 3);
    glUniform1i(glGetUniformLocation(default_program_, "u_emissive_texture"), 4);
    glUniform1i(glGetUniformLocation(default_program_, "u_environment_texture"), 5);
    glUseProgram(0);
}

void GLRasterizerScene::UploadFrameUniforms(const RenderSceneSnapshot& scene,
                                            const RenderViewSnapshot& view) {
    const RenderCameraSnapshot& camera = view.camera;
    const RenderEnvironmentSnapshot& environment = scene.environment;
    glUniformMatrix4fv(default_uniforms_.view_projection, 1, GL_FALSE, camera.view_projection.data());
    glUniformMatrix4fv(default_uniforms_.view, 1, GL_FALSE, camera.view.data());
    glUniform3f(default_uniforms_.camera_position,
                static_cast<float>(camera.world_position.x()),
                static_cast<float>(camera.world_position.y()),
                static_cast<float>(camera.world_position.z()));
    glUniform3f(default_uniforms_.light_direction,
                static_cast<float>(environment.directional_light_direction.x()),
                static_cast<float>(environment.directional_light_direction.y()),
                static_cast<float>(environment.directional_light_direction.z()));
    glUniform3f(default_uniforms_.light_color,
                environment.directional_light_color.red(),
                environment.directional_light_color.green(),
                environment.directional_light_color.blue());
    glUniform1f(default_uniforms_.light_intensity, static_cast<float>(environment.directional_light_intensity));
    glUniform3f(default_uniforms_.sky_color,
                environment.sky_color.red(),
                environment.sky_color.green(),
                environment.sky_color.blue());
    glUniform3f(default_uniforms_.ground_color,
                environment.ground_color.red(),
                environment.ground_color.green(),
                environment.ground_color.blue());
    glUniform1f(default_uniforms_.ambient_intensity, static_cast<float>(environment.ambient_intensity));
    glUniform1f(default_uniforms_.exposure, static_cast<float>(environment.exposure));
    glUniformMatrix3fv(default_uniforms_.environment_rotation,
                       1,
                       GL_FALSE,
                       environment.environment_rotation.data());
    glUniform1f(default_uniforms_.environment_intensity,
                static_cast<float>(environment.environment_intensity));
    const GLuint environment_texture = GetOrCreateTexture(environment.environment_texture);
    glBindTextureUnit(5, environment_texture);
    glUniform1i(default_uniforms_.has_environment_texture, environment_texture != 0);

    const std::size_t light_count = std::min(scene.lights.size(), default_uniforms_.light_position_type.size());
    glUniform1i(default_uniforms_.light_count, static_cast<GLint>(light_count));
    for (std::size_t i = 0; i < light_count; ++i) {
        const RenderLightSnapshot& light = scene.lights[i];
        glUniform4f(default_uniforms_.light_position_type[i],
                    static_cast<float>(light.position.x()),
                    static_cast<float>(light.position.y()),
                    static_cast<float>(light.position.z()),
                    static_cast<float>(light.type));
        glUniform4f(default_uniforms_.light_direction_range[i],
                    static_cast<float>(light.direction.x()),
                    static_cast<float>(light.direction.y()),
                    static_cast<float>(light.direction.z()),
                    static_cast<float>(light.range));
        glUniform4f(default_uniforms_.light_color_intensity[i],
                    light.color.red(),
                    light.color.green(),
                    light.color.blue(),
                    static_cast<float>(light.intensity));
        const RealType inner_radians = light.inner_angle * Math_PI / 180.0;
        const RealType outer_radians = light.outer_angle * Math_PI / 180.0;
        glUniform2f(default_uniforms_.light_spot_cosines[i],
                    static_cast<float>(std::cos(inner_radians)),
                    static_cast<float>(std::cos(outer_radians)));
    }
}

GLRasterizerScene::MeshCacheEntry* GLRasterizerScene::GetOrCreateMesh(const VisualMeshRenderItem& item) {
    const MeshSurfaceData* surface = item.GetSurface();
    if (surface == nullptr || surface->vertices.empty() || surface->indices.empty()) {
        return nullptr;
    }
    const MeshCacheKey key{item.mesh_id.operator std::uint64_t(), item.mesh_revision, item.surface_index};
    auto [it, inserted] = mesh_cache_.try_emplace(key);
    MeshCacheEntry& entry = it->second;
    entry.last_used_frame = frame_index_;
    if (!inserted) {
        return &entry;
    }

    std::vector<float> vertices;
    std::vector<float> normals;
    PackVector3(surface->vertices, &vertices);
    PackVector3(surface->normals, &normals);

    std::vector<float> colors;
    colors.reserve(surface->vertices.size() * 4);
    for (std::size_t i = 0; i < surface->vertices.size(); ++i) {
        const Color color = surface->colors.size() == surface->vertices.size()
                                    ? surface->colors[i]
                                    : Color{1.0f, 1.0f, 1.0f, 1.0f};
        colors.insert(colors.end(), {color.red(), color.green(), color.blue(), color.alpha()});
    }

    std::vector<float> tangents;
    tangents.reserve(surface->vertices.size() * 4);
    for (std::size_t i = 0; i < surface->vertices.size(); ++i) {
        const Vector4 tangent = surface->tangents.size() == surface->vertices.size()
                                        ? surface->tangents[i]
                                        : Vector4{1.0, 0.0, 0.0, 1.0};
        tangents.insert(tangents.end(),
                        {static_cast<float>(tangent.x()),
                         static_cast<float>(tangent.y()),
                         static_cast<float>(tangent.z()),
                         static_cast<float>(tangent.w())});
    }

    std::vector<float> uv;
    uv.reserve(surface->vertices.size() * 2);
    for (std::size_t i = 0; i < surface->vertices.size(); ++i) {
        const Vector2 value = surface->uv0.size() == surface->vertices.size()
                                      ? surface->uv0[i]
                                      : Vector2::Zero();
        uv.insert(uv.end(), {static_cast<float>(value.x()), static_cast<float>(value.y())});
    }

    glCreateVertexArrays(1, &entry.vao);
    glCreateBuffers(1, &entry.vertex_buffer);
    glCreateBuffers(1, &entry.normal_buffer);
    glCreateBuffers(1, &entry.color_buffer);
    glCreateBuffers(1, &entry.tangent_buffer);
    glCreateBuffers(1, &entry.uv_buffer);
    glCreateBuffers(1, &entry.index_buffer);
    glNamedBufferData(entry.vertex_buffer, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glNamedBufferData(entry.normal_buffer, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glNamedBufferData(entry.color_buffer, colors.size() * sizeof(float), colors.data(), GL_STATIC_DRAW);
    glNamedBufferData(entry.tangent_buffer, tangents.size() * sizeof(float), tangents.data(), GL_STATIC_DRAW);
    glNamedBufferData(entry.uv_buffer, uv.size() * sizeof(float), uv.data(), GL_STATIC_DRAW);
    glNamedBufferData(entry.index_buffer,
                      surface->indices.size() * sizeof(std::uint32_t),
                      surface->indices.data(),
                      GL_STATIC_DRAW);

    const std::array<std::pair<GLuint, GLuint>, 5> attributes = {
            std::pair{entry.vertex_buffer, 3U},
            std::pair{entry.normal_buffer, 3U},
            std::pair{entry.color_buffer, 4U},
            std::pair{entry.tangent_buffer, 4U},
            std::pair{entry.uv_buffer, 2U}};
    for (GLuint location = 0; location < attributes.size(); ++location) {
        const auto [buffer, components] = attributes[location];
        glVertexArrayVertexBuffer(entry.vao, location, buffer, 0, components * sizeof(float));
        glEnableVertexArrayAttrib(entry.vao, location);
        glVertexArrayAttribFormat(entry.vao, location, components, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribBinding(entry.vao, location, location);
    }
    glVertexArrayElementBuffer(entry.vao, entry.index_buffer);
    entry.index_count = static_cast<GLsizei>(surface->indices.size());
    return &entry;
}

GLuint GLRasterizerScene::GetOrCreateTexture(const RenderTextureSnapshot& texture) {
    if (!texture.IsValid()) {
        return 0;
    }
    const TextureCacheKey key{
            texture.texture_id.operator std::uint64_t(),
            texture.revision,
            texture.image.image_id.operator std::uint64_t(),
            texture.image.revision};
    auto [it, inserted] = texture_cache_.try_emplace(key);
    TextureCacheEntry& entry = it->second;
    entry.last_used_frame = frame_index_;
    if (!inserted) {
        return entry.texture;
    }

    const ImageStorageData& image = *texture.image.storage;
    GLenum internal_format = 0;
    GLenum pixel_format = 0;
    GLenum pixel_type = 0;
    if (image.data.empty() ||
        !ResolveImageFormat(image.format, &internal_format, &pixel_format, &pixel_type)) {
        LOG_WARN("OpenGL PBR texture format '{}' is not supported.", Image::GetFormatName(image.format));
        texture_cache_.erase(it);
        return 0;
    }

    glCreateTextures(GL_TEXTURE_2D, 1, &entry.texture);
    glTextureStorage2D(entry.texture, 1, internal_format, image.width, image.height);
    GLint old_unpack_alignment = 0;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &old_unpack_alignment);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTextureSubImage2D(entry.texture,
                        0,
                        0,
                        0,
                        image.width,
                        image.height,
                        pixel_format,
                        pixel_type,
                        image.data.data());
    glPixelStorei(GL_UNPACK_ALIGNMENT, old_unpack_alignment);
    glTextureParameteri(entry.texture,
                        GL_TEXTURE_MIN_FILTER,
                        texture.min_filter == TextureFilter::Nearest ? GL_NEAREST : GL_LINEAR);
    glTextureParameteri(entry.texture,
                        GL_TEXTURE_MAG_FILTER,
                        texture.mag_filter == TextureFilter::Nearest ? GL_NEAREST : GL_LINEAR);
    glTextureParameteri(entry.texture, GL_TEXTURE_WRAP_S, ToGLWrap(texture.wrap_u));
    glTextureParameteri(entry.texture, GL_TEXTURE_WRAP_T, ToGLWrap(texture.wrap_v));
    return entry.texture;
}

void GLRasterizerScene::DrawVisualItem(const VisualMeshRenderItem& item) {
    MeshCacheEntry* mesh = GetOrCreateMesh(item);
    if (mesh == nullptr || mesh->index_count <= 0) {
        return;
    }

    const RenderMaterialSnapshot& material = item.material;
    if (material.alpha_mode == AlphaMode::Blend) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
    } else {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
    if (material.double_sided) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
    }

    const Matrix3 linear = item.model.template block<3, 3>(0, 0);
    Matrix3 normal_matrix = Matrix3::Identity();
    if (std::abs(linear.determinant()) > CMP_EPSILON) {
        normal_matrix = linear.inverse().transpose();
    }
    glUniformMatrix4fv(default_uniforms_.model, 1, GL_FALSE, item.model.data());
    glUniformMatrix3fv(default_uniforms_.normal_matrix, 1, GL_FALSE, normal_matrix.data());
    glUniform1ui(default_uniforms_.instance_id, item.instance_id);
    glUniform1ui(default_uniforms_.semantic_id, item.semantic_id);
    glUniform4f(default_uniforms_.color,
                material.albedo.red(),
                material.albedo.green(),
                material.albedo.blue(),
                material.albedo.alpha());
    glUniform1f(default_uniforms_.metallic, static_cast<float>(material.metallic));
    glUniform1f(default_uniforms_.roughness, static_cast<float>(material.roughness));
    glUniform1f(default_uniforms_.specular, static_cast<float>(material.specular));
    glUniform1f(default_uniforms_.normal_scale, static_cast<float>(material.normal_scale));
    glUniform1f(default_uniforms_.occlusion_strength, static_cast<float>(material.occlusion_strength));
    glUniform3f(default_uniforms_.emissive,
                material.emissive.red(),
                material.emissive.green(),
                material.emissive.blue());
    glUniform1i(default_uniforms_.alpha_mode, static_cast<int>(material.alpha_mode));
    glUniform1f(default_uniforms_.alpha_cutoff, static_cast<float>(material.alpha_cutoff));

    const std::array<RenderTextureSnapshot, 5> textures = {
            material.albedo_texture,
            material.metallic_roughness_texture,
            material.normal_texture,
            material.occlusion_texture,
            material.emissive_texture};
    std::array<GLuint, 5> handles{};
    for (std::size_t i = 0; i < textures.size(); ++i) {
        handles[i] = GetOrCreateTexture(textures[i]);
        glBindTextureUnit(static_cast<GLuint>(i), handles[i]);
    }
    glUniform1i(default_uniforms_.has_albedo_texture, handles[0] != 0);
    glUniform1i(default_uniforms_.has_metallic_roughness_texture, handles[1] != 0);
    glUniform1i(default_uniforms_.has_normal_texture, handles[2] != 0);
    glUniform1i(default_uniforms_.has_occlusion_texture, handles[3] != 0);
    glUniform1i(default_uniforms_.has_emissive_texture, handles[4] != 0);

    glBindVertexArray(mesh->vao);
    glDrawElements(GL_TRIANGLES, mesh->index_count, GL_UNSIGNED_INT, nullptr);
}

void GLRasterizerScene::PruneCaches() {
    if (frame_index_ % 120 != 0) {
        return;
    }
    constexpr std::uint64_t keep_frames = 360;
    for (auto it = mesh_cache_.begin(); it != mesh_cache_.end();) {
        if (it->second.last_used_frame + keep_frames < frame_index_) {
            DestroyMeshEntry(it->second);
            it = mesh_cache_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = texture_cache_.begin(); it != texture_cache_.end();) {
        if (it->second.last_used_frame + keep_frames < frame_index_) {
            if (it->second.texture != 0) {
                glDeleteTextures(1, &it->second.texture);
            }
            it = texture_cache_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace gobot::opengl
