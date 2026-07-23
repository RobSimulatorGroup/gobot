#include "gobot/drivers/opengl/rasterizer_scene_gl.hpp"

#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/log.hpp"
#include "glsl_shader_hpp/fullscreen_triangle_vert.hpp"
#include "glsl_shader_hpp/fxaa_frag.hpp"
#include "glsl_shader_hpp/shadow_depth_frag.hpp"
#include "glsl_shader_hpp/shadow_depth_vert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace gobot::opengl {
namespace {

GLuint CompilePassShader(GLenum type, const char* source, const char* name) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_TRUE) {
        return shader;
    }

    GLsizei log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::string log(static_cast<std::size_t>(std::max(log_length, GLsizei{1})), '\0');
    glGetShaderInfoLog(shader, log_length, &log_length, log.data());
    LOG_ERROR("{} shader compilation failed: {}", name, log);
    glDeleteShader(shader);
    return 0;
}

GLuint LinkPassProgram(const char* vertex_source, const char* fragment_source, const char* name) {
    const GLuint vertex = CompilePassShader(GL_VERTEX_SHADER, vertex_source, name);
    const GLuint fragment = CompilePassShader(GL_FRAGMENT_SHADER, fragment_source, name);
    if (vertex == 0 || fragment == 0) {
        if (vertex != 0) glDeleteShader(vertex);
        if (fragment != 0) glDeleteShader(fragment);
        return 0;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_TRUE) {
        return program;
    }

    GLsizei log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    std::string log(static_cast<std::size_t>(std::max(log_length, GLsizei{1})), '\0');
    glGetProgramInfoLog(program, log_length, &log_length, log.data());
    LOG_ERROR("{} program link failed: {}", name, log);
    glDeleteProgram(program);
    return 0;
}

std::pair<int, int> ShadowQualityParameters(RasterShadowQuality quality) {
    switch (quality) {
        case RasterShadowQuality::Low: return {1024, 0};
        case RasterShadowQuality::Medium: return {2048, 1};
        case RasterShadowQuality::High: return {4096, 2};
        case RasterShadowQuality::Disabled: return {0, 0};
    }
    return {0, 0};
}

std::array<Vector3, 8> BoundsCorners(const AABB& bounds) {
    const Vector3 minimum = bounds.GetMin();
    const Vector3 maximum = bounds.GetMax();
    return {
            Vector3{minimum.x(), minimum.y(), minimum.z()},
            Vector3{maximum.x(), minimum.y(), minimum.z()},
            Vector3{minimum.x(), maximum.y(), minimum.z()},
            Vector3{maximum.x(), maximum.y(), minimum.z()},
            Vector3{minimum.x(), minimum.y(), maximum.z()},
            Vector3{maximum.x(), minimum.y(), maximum.z()},
            Vector3{minimum.x(), maximum.y(), maximum.z()},
            Vector3{maximum.x(), maximum.y(), maximum.z()}};
}

bool BuildShadowMatrix(const RenderViewSnapshot& view,
                       const RenderDrawLists& draw_lists,
                       const RenderLightSnapshot& light,
                       RealType shadow_distance,
                       int resolution,
                       Matrix4* result) {
    const Matrix4& view_projection = view.camera.view_projection;
    if (!view_projection.allFinite() || std::abs(view_projection.determinant()) <= CMP_EPSILON) {
        return false;
    }

    const Matrix4 inverse = view_projection.inverse();
    std::array<Vector3, 8> frustum_corners{};
    std::size_t corner_index = 0;
    for (const RealType z : {-1.0, 1.0}) {
        for (const RealType y : {-1.0, 1.0}) {
            for (const RealType x : {-1.0, 1.0}) {
                Vector4 world = inverse * Vector4{x, y, z, 1.0};
                if (!world.allFinite() || std::abs(world.w()) <= CMP_EPSILON) {
                    return false;
                }
                world /= world.w();
                frustum_corners[corner_index++] = world.template head<3>();
            }
        }
    }

    const RealType range = std::max(view.camera.z_far - view.camera.z_near,
                                    static_cast<RealType>(CMP_EPSILON));
    const RealType clamped_distance = std::clamp(
            shadow_distance, view.camera.z_near, view.camera.z_far);
    const RealType far_fraction = (clamped_distance - view.camera.z_near) / range;
    for (std::size_t i = 0; i < 4; ++i) {
        frustum_corners[4 + i] = frustum_corners[i] +
                                 (frustum_corners[4 + i] - frustum_corners[i]) * far_fraction;
    }

    Vector3 center = Vector3::Zero();
    for (const Vector3& corner : frustum_corners) center += corner;
    center /= static_cast<RealType>(frustum_corners.size());

    Vector3 light_direction = light.direction;
    if (!light_direction.allFinite() || light_direction.norm() <= CMP_EPSILON) {
        return false;
    }
    light_direction.normalize();
    const Vector3 up = std::abs(light_direction.dot(Vector3::UnitZ())) > 0.95
                               ? Vector3::UnitY()
                               : Vector3::UnitZ();

    RealType receiver_radius = 1.0;
    for (const Vector3& corner : frustum_corners) {
        receiver_radius = std::max(receiver_radius, (corner - center).norm());
    }
    const Matrix4 light_view = Matrix4::LookAt(
            center + light_direction * receiver_radius * 2.0, center, up);

    Vector3 minimum = Vector3::Constant(std::numeric_limits<RealType>::infinity());
    Vector3 maximum = Vector3::Constant(-std::numeric_limits<RealType>::infinity());
    const auto include_point = [&](const Vector3& point, Vector3* min_value, Vector3* max_value) {
        const Vector4 transformed = light_view * Vector4{point.x(), point.y(), point.z(), 1.0};
        *min_value = min_value->cwiseMin(transformed.template head<3>());
        *max_value = max_value->cwiseMax(transformed.template head<3>());
    };
    for (const Vector3& corner : frustum_corners) include_point(corner, &minimum, &maximum);

    const RealType margin_x = std::max<RealType>(0.1, (maximum.x() - minimum.x()) * 0.05);
    const RealType margin_y = std::max<RealType>(0.1, (maximum.y() - minimum.y()) * 0.05);
    minimum.x() -= margin_x;
    maximum.x() += margin_x;
    minimum.y() -= margin_y;
    maximum.y() += margin_y;

    for (const PreparedRenderItem& prepared : draw_lists.shadow_casters) {
        if (!prepared.item->world_bounds.IsValid()) {
            continue;
        }
        const auto corners = BoundsCorners(prepared.item->world_bounds);
        Vector3 caster_min = Vector3::Constant(std::numeric_limits<RealType>::infinity());
        Vector3 caster_max = Vector3::Constant(-std::numeric_limits<RealType>::infinity());
        for (const Vector3& corner : corners) include_point(corner, &caster_min, &caster_max);
        const bool overlaps_receiver = caster_max.x() >= minimum.x() && caster_min.x() <= maximum.x() &&
                                       caster_max.y() >= minimum.y() && caster_min.y() <= maximum.y();
        if (overlaps_receiver) {
            minimum.z() = std::min(minimum.z(), caster_min.z());
            maximum.z() = std::max(maximum.z(), caster_max.z());
        }
    }

    const RealType width = std::max(maximum.x() - minimum.x(), static_cast<RealType>(0.01));
    const RealType height = std::max(maximum.y() - minimum.y(), static_cast<RealType>(0.01));
    const RealType texel_x = width / static_cast<RealType>(resolution);
    const RealType texel_y = height / static_cast<RealType>(resolution);
    RealType center_x = (minimum.x() + maximum.x()) * 0.5;
    RealType center_y = (minimum.y() + maximum.y()) * 0.5;
    center_x = std::round(center_x / texel_x) * texel_x;
    center_y = std::round(center_y / texel_y) * texel_y;
    minimum.x() = center_x - width * 0.5;
    maximum.x() = center_x + width * 0.5;
    minimum.y() = center_y - height * 0.5;
    maximum.y() = center_y + height * 0.5;

    const RealType depth_padding = std::max<RealType>(1.0, receiver_radius * 0.1);
    const RealType z_near = -maximum.z() - depth_padding;
    const RealType z_far = std::max(z_near + static_cast<RealType>(0.01),
                                    -minimum.z() + depth_padding);
    Matrix4 light_projection = Matrix4::Ortho(
            minimum.x(), maximum.x(), minimum.y(), maximum.y(), z_near, z_far);
    Matrix4 light_view_projection = light_projection * light_view;
    Vector4 shadow_origin = light_view_projection * Vector4{0.0, 0.0, 0.0, 1.0};
    if (std::abs(shadow_origin.w()) > CMP_EPSILON) {
        shadow_origin /= shadow_origin.w();
        const Vector2 texel_origin = shadow_origin.template head<2>() *
                                     (static_cast<RealType>(resolution) * 0.5);
        const Vector2 rounded_origin = texel_origin.array().round().matrix();
        const Vector2 offset = (rounded_origin - texel_origin) *
                               (2.0 / static_cast<RealType>(resolution));
        light_projection(0, 3) += offset.x();
        light_projection(1, 3) += offset.y();
        light_view_projection = light_projection * light_view;
    }
    *result = light_view_projection;
    return result->allFinite();
}

} // namespace

bool GLRasterizerScene::EnsureShadowPassResources(int resolution) {
    if (shadow_pass_.program == 0) {
        shadow_pass_.program = LinkPassProgram(
                SHADOW_DEPTH_VERT, SHADOW_DEPTH_FRAG, "Directional shadow");
        if (shadow_pass_.program == 0) return false;
        shadow_pass_.model = glGetUniformLocation(shadow_pass_.program, "u_model");
        shadow_pass_.light_view_projection =
                glGetUniformLocation(shadow_pass_.program, "u_light_view_projection");
        shadow_pass_.alpha = glGetUniformLocation(shadow_pass_.program, "u_alpha");
        shadow_pass_.alpha_cutoff = glGetUniformLocation(shadow_pass_.program, "u_alpha_cutoff");
        shadow_pass_.has_albedo_texture =
                glGetUniformLocation(shadow_pass_.program, "u_has_albedo_texture");
        glUseProgram(shadow_pass_.program);
        glUniform1i(glGetUniformLocation(shadow_pass_.program, "u_albedo_texture"), 0);
        glUseProgram(0);
    }
    if (shadow_pass_.framebuffer == 0) {
        glCreateFramebuffers(1, &shadow_pass_.framebuffer);
        glNamedFramebufferDrawBuffer(shadow_pass_.framebuffer, GL_NONE);
        glNamedFramebufferReadBuffer(shadow_pass_.framebuffer, GL_NONE);
    }
    if (shadow_pass_.resolution != resolution || shadow_pass_.depth_texture == 0) {
        if (shadow_pass_.depth_texture != 0) {
            glDeleteTextures(1, &shadow_pass_.depth_texture);
        }
        glCreateTextures(GL_TEXTURE_2D, 1, &shadow_pass_.depth_texture);
        glTextureStorage2D(shadow_pass_.depth_texture, 1, GL_DEPTH_COMPONENT32F, resolution, resolution);
        glTextureParameteri(shadow_pass_.depth_texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(shadow_pass_.depth_texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(shadow_pass_.depth_texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTextureParameteri(shadow_pass_.depth_texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTextureParameteri(shadow_pass_.depth_texture, GL_TEXTURE_COMPARE_MODE, GL_NONE);
        const std::array<float, 4> border{1.0f, 1.0f, 1.0f, 1.0f};
        glTextureParameterfv(shadow_pass_.depth_texture, GL_TEXTURE_BORDER_COLOR, border.data());
        glNamedFramebufferTexture(
                shadow_pass_.framebuffer, GL_DEPTH_ATTACHMENT, shadow_pass_.depth_texture, 0);
        shadow_pass_.resolution = resolution;
    }
    return glCheckNamedFramebufferStatus(shadow_pass_.framebuffer, GL_FRAMEBUFFER) ==
           GL_FRAMEBUFFER_COMPLETE;
}

bool GLRasterizerScene::RenderDirectionalShadow(const RenderSceneSnapshot& scene,
                                                const RenderViewSnapshot& view,
                                                const RenderDrawLists& draw_lists) {
    shadow_pass_.active = false;
    shadow_pass_.light_index = -1;
    const auto [resolution, filter_radius] = ShadowQualityParameters(
            settings_.raster.shadow_quality);
    if (resolution == 0 || draw_lists.shadow_casters.empty()) {
        return false;
    }

    const std::size_t light_limit = std::min<std::size_t>(scene.lights.size(), 16);
    for (std::size_t index = 0; index < light_limit; ++index) {
        const RenderLightSnapshot& light = scene.lights[index];
        if (light.type != RenderLightType::Directional || !light.shadow_enabled) continue;
        if (!EnsureShadowPassResources(resolution) ||
            !BuildShadowMatrix(view,
                               draw_lists,
                               light,
                               settings_.raster.shadow_distance,
                               resolution,
                               &shadow_pass_.view_projection)) {
            return false;
        }
        shadow_pass_.light_index = static_cast<int>(index);
        shadow_pass_.bias = light.shadow_bias;
        shadow_pass_.normal_bias = light.shadow_normal_bias;
        shadow_pass_.filter_radius = filter_radius;
        break;
    }
    if (shadow_pass_.light_index < 0) return false;

    glBindFramebuffer(GL_FRAMEBUFFER, shadow_pass_.framebuffer);
    glViewport(0, 0, resolution, resolution);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glUseProgram(shadow_pass_.program);
    glUniformMatrix4fv(shadow_pass_.light_view_projection,
                       1,
                       GL_FALSE,
                       shadow_pass_.view_projection.data());
    for (const PreparedRenderItem& prepared : draw_lists.shadow_casters) {
        stats_.shadow_draw_calls += DrawShadowItem(*prepared.item) ? 1u : 0u;
    }
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    shadow_pass_.active = true;
    return true;
}

bool GLRasterizerScene::DrawShadowItem(const VisualMeshRenderItem& item) {
    MeshCacheEntry* mesh = GetOrCreateMesh(item);
    if (mesh == nullptr || mesh->index_count <= 0) return false;

    const RenderMaterialSnapshot& material = item.material;
    if (material.double_sided) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
    }
    glUniformMatrix4fv(shadow_pass_.model, 1, GL_FALSE, item.model.data());
    glUniform1f(shadow_pass_.alpha, material.albedo.alpha());
    glUniform1f(shadow_pass_.alpha_cutoff,
                material.alpha_mode == AlphaMode::Mask
                        ? static_cast<float>(material.alpha_cutoff)
                        : -1.0f);
    const GLuint albedo_texture = material.alpha_mode == AlphaMode::Mask
                                          ? GetOrCreateTexture(material.albedo_texture)
                                          : 0;
    glBindTextureUnit(0, albedo_texture);
    glUniform1i(shadow_pass_.has_albedo_texture, albedo_texture != 0);
    glBindVertexArray(mesh->vao);
    glDrawElements(GL_TRIANGLES, mesh->index_count, GL_UNSIGNED_INT, nullptr);
    return true;
}

void GLRasterizerScene::UploadShadowUniforms() {
    glUniform1i(default_uniforms_.has_shadow_map, shadow_pass_.active ? 1 : 0);
    glUniform1i(default_uniforms_.shadow_light_index,
                shadow_pass_.active ? shadow_pass_.light_index : -1);
    if (!shadow_pass_.active) return;

    glBindTextureUnit(6, shadow_pass_.depth_texture);
    glUniformMatrix4fv(default_uniforms_.shadow_view_projection,
                       1,
                       GL_FALSE,
                       shadow_pass_.view_projection.data());
    glUniform1f(default_uniforms_.shadow_bias, static_cast<float>(shadow_pass_.bias));
    glUniform1f(default_uniforms_.shadow_normal_bias,
                static_cast<float>(shadow_pass_.normal_bias));
    const float inverse_resolution = 1.0f / static_cast<float>(shadow_pass_.resolution);
    glUniform2f(default_uniforms_.shadow_texel_size, inverse_resolution, inverse_resolution);
    glUniform1i(default_uniforms_.shadow_filter_radius, shadow_pass_.filter_radius);
}

bool GLRasterizerScene::EnsureFxaaPassResources(int width, int height, GLuint depth_texture) {
    if (fxaa_pass_.program == 0) {
        fxaa_pass_.program = LinkPassProgram(
                FULLSCREEN_TRIANGLE_VERT, FXAA_FRAG, "FXAA");
        if (fxaa_pass_.program == 0) return false;
        fxaa_pass_.inverse_size = glGetUniformLocation(fxaa_pass_.program, "u_inverse_size");
        glUseProgram(fxaa_pass_.program);
        glUniform1i(glGetUniformLocation(fxaa_pass_.program, "u_color_texture"), 0);
        glUseProgram(0);
        glCreateVertexArrays(1, &fxaa_pass_.vertex_array);
    }
    if (fxaa_pass_.framebuffer == 0) {
        glCreateFramebuffers(1, &fxaa_pass_.framebuffer);
        glNamedFramebufferDrawBuffer(fxaa_pass_.framebuffer, GL_COLOR_ATTACHMENT0);
    }
    if (fxaa_pass_.width != width || fxaa_pass_.height != height || fxaa_pass_.color_texture == 0) {
        if (fxaa_pass_.color_texture != 0) {
            glDeleteTextures(1, &fxaa_pass_.color_texture);
        }
        glCreateTextures(GL_TEXTURE_2D, 1, &fxaa_pass_.color_texture);
        glTextureStorage2D(fxaa_pass_.color_texture, 1, GL_RGBA8, width, height);
        glTextureParameteri(fxaa_pass_.color_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(fxaa_pass_.color_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(fxaa_pass_.color_texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(fxaa_pass_.color_texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glNamedFramebufferTexture(
                fxaa_pass_.framebuffer, GL_COLOR_ATTACHMENT0, fxaa_pass_.color_texture, 0);
        fxaa_pass_.width = width;
        fxaa_pass_.height = height;
    }
    glNamedFramebufferTexture(fxaa_pass_.framebuffer, GL_DEPTH_ATTACHMENT, depth_texture, 0);
    return glCheckNamedFramebufferStatus(fxaa_pass_.framebuffer, GL_FRAMEBUFFER) ==
           GL_FRAMEBUFFER_COMPLETE;
}

void GLRasterizerScene::ApplyFxaa(const RenderTarget& target) {
    glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
    glViewport(0, 0, target.size.x(), target.size.y());
    ConfigureDrawBuffers(target, true);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glUseProgram(fxaa_pass_.program);
    glBindTextureUnit(0, fxaa_pass_.color_texture);
    glUniform2f(fxaa_pass_.inverse_size,
                1.0f / static_cast<float>(target.size.x()),
                1.0f / static_cast<float>(target.size.y()));
    glBindVertexArray(fxaa_pass_.vertex_array);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glUseProgram(0);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    ConfigureDrawBuffers(target, false);
}

void GLRasterizerScene::ConfigureDrawBuffers(const RenderTarget& target, bool rgb_only) {
    if (rgb_only) {
        const GLenum color = GL_COLOR_ATTACHMENT0;
        glDrawBuffers(1, &color);
        return;
    }
    std::array<GLenum, 5> draw_buffers{};
    draw_buffers.fill(GL_NONE);
    for (std::uint32_t index = 0; index < draw_buffers.size(); ++index) {
        const auto output = static_cast<RenderOutputType>(index);
        if (RenderOutputMaskContains(target.output_mask, output)) {
            draw_buffers[index] = GL_COLOR_ATTACHMENT0 + index;
        }
    }
    glDrawBuffers(static_cast<GLsizei>(draw_buffers.size()), draw_buffers.data());
}

void GLRasterizerScene::DestroyPassResources() {
    if (shadow_pass_.program != 0) glDeleteProgram(shadow_pass_.program);
    if (shadow_pass_.framebuffer != 0) glDeleteFramebuffers(1, &shadow_pass_.framebuffer);
    if (shadow_pass_.depth_texture != 0) glDeleteTextures(1, &shadow_pass_.depth_texture);
    if (fxaa_pass_.program != 0) glDeleteProgram(fxaa_pass_.program);
    if (fxaa_pass_.framebuffer != 0) glDeleteFramebuffers(1, &fxaa_pass_.framebuffer);
    if (fxaa_pass_.color_texture != 0) glDeleteTextures(1, &fxaa_pass_.color_texture);
    if (fxaa_pass_.vertex_array != 0) glDeleteVertexArrays(1, &fxaa_pass_.vertex_array);
    shadow_pass_ = {};
    fxaa_pass_ = {};
}

} // namespace gobot::opengl
