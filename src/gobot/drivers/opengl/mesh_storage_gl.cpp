/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-8-9.
*/

#include "gobot/drivers/opengl/mesh_storage_gl.hpp"

#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/mesh_instance_3d.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"

#include <array>

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

GLMeshStorage* GLMeshStorage::s_singleton = nullptr;

MeshStorage* GLMeshStorage::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize MeshStorage");
    return s_singleton;
}

GLMeshStorage::GLMeshStorage() {
    s_singleton = this;
}

GLMeshStorage::~GLMeshStorage() {
    if (default_program_ != 0) {
        glDeleteProgram(default_program_);
        default_program_ = 0;
    }
    s_singleton = nullptr;
}


RID GLMeshStorage::MeshAllocate() {
    return mesh_owner_.AllocateRID();
}

void GLMeshStorage::MeshInitialize(const RID& p_rid)  {
    mesh_owner_.InitializeRID(p_rid, MeshData());
}

void GLMeshStorage::MeshSetBox(const RID& p_rid, const Vector3& size) {
    MeshData* mesh = mesh_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(mesh == nullptr);

    const Vector3 half = size * 0.5f;
    const std::array<Vector3, 8> p = {
            Vector3{-half.x(), -half.y(), -half.z()},
            Vector3{ half.x(), -half.y(), -half.z()},
            Vector3{ half.x(),  half.y(), -half.z()},
            Vector3{-half.x(),  half.y(), -half.z()},
            Vector3{-half.x(), -half.y(),  half.z()},
            Vector3{ half.x(), -half.y(),  half.z()},
            Vector3{ half.x(),  half.y(),  half.z()},
            Vector3{-half.x(),  half.y(),  half.z()},
    };

    constexpr std::array<uint32_t, 36> indices = {
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            3, 6, 2, 3, 7, 6,
            1, 2, 6, 1, 6, 5,
            0, 4, 7, 0, 7, 3,
    };

    mesh->vertices.clear();
    mesh->vertices.reserve(p.size() * 3);
    for (const Vector3& point : p) {
        mesh->vertices.push_back(point.x());
        mesh->vertices.push_back(point.y());
        mesh->vertices.push_back(point.z());
    }
    mesh->indices.assign(indices.begin(), indices.end());
    mesh->index_count = static_cast<GLsizei>(mesh->indices.size());
    mesh->dirty = true;
}

bool GLMeshStorage::OwnsMesh(const RID& p_rid) const {
    return mesh_owner_.Owns(p_rid);
}

void GLMeshStorage::MeshFree(const RID& p_rid) {
    MeshData* mesh = mesh_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(mesh == nullptr);
    if (mesh->vao != 0) {
        glDeleteVertexArrays(1, &mesh->vao);
    }
    if (mesh->vertex_buffer != 0) {
        glDeleteBuffers(1, &mesh->vertex_buffer);
    }
    if (mesh->index_buffer != 0) {
        glDeleteBuffers(1, &mesh->index_buffer);
    }
    mesh_owner_.Free(p_rid);
}

void GLMeshStorage::EnsureDefaultProgram() {
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

void GLMeshStorage::UploadMesh(MeshData* mesh) {
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

void GLMeshStorage::RenderScene(const RID& render_target, const SceneTree* scene_tree, const Camera3D* camera) {
    ERR_FAIL_COND(scene_tree == nullptr);
    ERR_FAIL_COND(camera == nullptr);

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
    glClearColor(0.07f, 0.075f, 0.085f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(default_program_);
    const Matrix4 view = camera->GetViewMatrix();
    const Matrix4 projection = camera->GetProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(default_program_, "u_view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(default_program_, "u_projection"), 1, GL_FALSE, projection.data());
    glUniform4f(glGetUniformLocation(default_program_, "u_color"), 0.66f, 0.78f, 0.95f, 1.0f);

    DrawNode(scene_tree->GetRoot(), view, projection);

    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLMeshStorage::DrawNode(Node* node, const Matrix4& view, const Matrix4& projection) {
    (void)view;
    (void)projection;

    auto* mesh_instance = Object::PointerCastTo<MeshInstance3D>(node);
    if (mesh_instance && mesh_instance->IsInsideTree() && mesh_instance->IsVisibleInTree()) {
        Ref<Mesh> mesh_resource = mesh_instance->GetMesh();
        if (mesh_resource.IsValid()) {
            RID mesh_rid = mesh_resource->GetRid();
            MeshData* mesh = mesh_owner_.GetOrNull(mesh_rid);
            if (mesh && mesh->index_count > 0) {
                UploadMesh(mesh);
                Matrix4 model = mesh_instance->GetGlobalTransform().matrix();
                glUniformMatrix4fv(glGetUniformLocation(default_program_, "u_model"), 1, GL_FALSE, model.data());
                glBindVertexArray(mesh->vao);
                glDrawElements(GL_TRIANGLES, mesh->index_count, GL_UNSIGNED_INT, nullptr);
            }
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        DrawNode(node->GetChild(static_cast<int>(i)), view, projection);
    }
}

}
