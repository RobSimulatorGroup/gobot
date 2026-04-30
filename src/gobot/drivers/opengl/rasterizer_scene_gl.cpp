/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-23
*/

#include "gobot/drivers/opengl/rasterizer_scene_gl.hpp"

#include "gobot/drivers/opengl/mesh_storage_gl.hpp"
#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/resources/material.hpp"

#include <algorithm>
#include <string>

namespace gobot::opengl {

namespace {

GLuint CompileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
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

}

GLRasterizerScene::GLRasterizerScene(GLMeshStorage* mesh_storage)
    : mesh_storage_(mesh_storage) {
}

GLRasterizerScene::~GLRasterizerScene() {
    if (default_program_ != 0) {
        glDeleteProgram(default_program_);
        default_program_ = 0;
    }
}

void GLRasterizerScene::RenderScene(const RID& render_target, const Node* scene_root, const Camera3D* camera) {
    ERR_FAIL_COND(scene_root == nullptr);
    ERR_FAIL_COND(camera == nullptr);
    ERR_FAIL_COND(mesh_storage_ == nullptr);

    auto* rt = TextureStorage::GetInstance()->GetRenderTarget(render_target);
    ERR_FAIL_COND(rt == nullptr);
    if (rt->fbo == 0 || rt->size.x() <= 0 || rt->size.y() <= 0) {
        return;
    }

    EnsureDefaultProgram();
    if (default_program_ == 0) {
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);
    glViewport(0, 0, rt->size.x(), rt->size.y());
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClearColor(0.11f, 0.115f, 0.125f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(default_program_);
    const Matrix4 view = camera->GetViewMatrix();
    const Matrix4 projection = camera->GetProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(default_program_, "u_view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(default_program_, "u_projection"), 1, GL_FALSE, projection.data());

    const SceneRenderItems render_items = CollectSceneRenderItems(scene_root);
    for (const VisualMeshRenderItem& item : render_items.visual_meshes) {
        DrawVisualItem(item);
    }

    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLRasterizerScene::EnsureDefaultProgram() {
    if (default_program_ != 0) {
        return;
    }

    static constexpr const char* vertex_shader = R"(
#version 460 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
)";

    static constexpr const char* fragment_shader = R"(
#version 460 core
out vec4 frag_color;
uniform vec4 u_color;
void main() {
    frag_color = u_color;
}
)";

    GLuint vs = CompileShader(GL_VERTEX_SHADER, vertex_shader);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fragment_shader);
    if (vs == 0 || fs == 0) {
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
    }
}

void GLRasterizerScene::UploadMesh(GLMeshData* mesh) {
    ERR_FAIL_COND(mesh == nullptr);
    if (!mesh->dirty) {
        return;
    }

    if (mesh->vao == 0) {
        glCreateVertexArrays(1, &mesh->vao);
    }
    if (mesh->vertex_buffer == 0) {
        glCreateBuffers(1, &mesh->vertex_buffer);
    }
    if (mesh->index_buffer == 0) {
        glCreateBuffers(1, &mesh->index_buffer);
    }

    glNamedBufferData(mesh->vertex_buffer,
                      static_cast<GLsizeiptr>(mesh->vertices.size() * sizeof(float)),
                      mesh->vertices.data(),
                      GL_STATIC_DRAW);
    glNamedBufferData(mesh->index_buffer,
                      static_cast<GLsizeiptr>(mesh->indices.size() * sizeof(uint32_t)),
                      mesh->indices.data(),
                      GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(mesh->vao, 0, mesh->vertex_buffer, 0, 3 * sizeof(float));
    glEnableVertexArrayAttrib(mesh->vao, 0);
    glVertexArrayAttribFormat(mesh->vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(mesh->vao, 0, 0);
    glVertexArrayElementBuffer(mesh->vao, mesh->index_buffer);

    mesh->dirty = false;
}

void GLRasterizerScene::DrawVisualItem(const VisualMeshRenderItem& item) {
    GLMeshData* mesh = mesh_storage_->mesh_owner_.GetOrNull(item.mesh);
    if (mesh == nullptr || mesh->index_count <= 0) {
        return;
    }

    UploadMesh(mesh);
    Color color = item.surface_color;
    if (Ref<PBRMaterial3D> pbr_material = dynamic_pointer_cast<PBRMaterial3D>(item.material);
        pbr_material.IsValid()) {
        color = pbr_material->GetAlbedo();
    }

    glUniformMatrix4fv(glGetUniformLocation(default_program_, "u_model"), 1, GL_FALSE, item.model.data());
    glUniform4f(glGetUniformLocation(default_program_, "u_color"),
                color.red(),
                color.green(),
                color.blue(),
                color.alpha());
    glBindVertexArray(mesh->vao);
    glDrawElements(GL_TRIANGLES, mesh->index_count, GL_UNSIGNED_INT, nullptr);
}

}
