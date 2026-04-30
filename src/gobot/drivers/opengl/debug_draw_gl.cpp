/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
*/

#include "gobot/drivers/opengl/debug_draw_gl.hpp"

#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/collision_shape_3d.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace gobot::opengl {

namespace {

GLuint CompileDebugShader(GLenum type, const char* source) {
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
        LOG_ERROR("Editor debug shader compilation failed: {}", log);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

void FreeLineBuffer(GLRendererDebugDraw::LineBuffer& buffer) {
    if (buffer.vao != 0) {
        glDeleteVertexArrays(1, &buffer.vao);
        buffer.vao = 0;
    }
    if (buffer.vertex_buffer != 0) {
        glDeleteBuffers(1, &buffer.vertex_buffer);
        buffer.vertex_buffer = 0;
    }
    buffer.vertex_count = 0;
}

void PushWorldVertex(std::vector<float>& vertices, const Vector3& point) {
    vertices.push_back(static_cast<float>(point.x()));
    vertices.push_back(static_cast<float>(point.y()));
    vertices.push_back(static_cast<float>(point.z()));
}

void AppendLine(std::vector<float>& vertices,
                const Affine3& transform,
                const Vector3& from,
                const Vector3& to) {
    PushWorldVertex(vertices, transform * from);
    PushWorldVertex(vertices, transform * to);
}

void AppendBoxLines(std::vector<float>& vertices, const Affine3& transform, const Vector3& size) {
    const Vector3 half = size * 0.5f;
    const std::array<Vector3, 8> corners = {
            Vector3{-half.x(), -half.y(), -half.z()},
            Vector3{ half.x(), -half.y(), -half.z()},
            Vector3{ half.x(),  half.y(), -half.z()},
            Vector3{-half.x(),  half.y(), -half.z()},
            Vector3{-half.x(), -half.y(),  half.z()},
            Vector3{ half.x(), -half.y(),  half.z()},
            Vector3{ half.x(),  half.y(),  half.z()},
            Vector3{-half.x(),  half.y(),  half.z()},
    };
    constexpr std::array<std::pair<int, int>, 12> edges = {
            std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}, std::pair{3, 0},
            std::pair{4, 5}, std::pair{5, 6}, std::pair{6, 7}, std::pair{7, 4},
            std::pair{0, 4}, std::pair{1, 5}, std::pair{2, 6}, std::pair{3, 7},
    };

    for (const auto& [from, to] : edges) {
        AppendLine(vertices, transform, corners[from], corners[to]);
    }
}

void AppendCircleLines(std::vector<float>& vertices,
                       const Affine3& transform,
                       RealType radius,
                       int segments,
                       int axis) {
    for (int i = 0; i < segments; ++i) {
        const RealType a = static_cast<RealType>(2.0 * Math_PI * i / segments);
        const RealType b = static_cast<RealType>(2.0 * Math_PI * ((i + 1) % segments) / segments);

        Vector3 from = Vector3::Zero();
        Vector3 to = Vector3::Zero();
        if (axis == 0) {
            from = Vector3{0.0, std::cos(a) * radius, std::sin(a) * radius};
            to = Vector3{0.0, std::cos(b) * radius, std::sin(b) * radius};
        } else if (axis == 1) {
            from = Vector3{std::cos(a) * radius, 0.0, std::sin(a) * radius};
            to = Vector3{std::cos(b) * radius, 0.0, std::sin(b) * radius};
        } else {
            from = Vector3{std::cos(a) * radius, std::sin(a) * radius, 0.0};
            to = Vector3{std::cos(b) * radius, std::sin(b) * radius, 0.0};
        }
        AppendLine(vertices, transform, from, to);
    }
}

void AppendSphereLines(std::vector<float>& vertices, const Affine3& transform, RealType radius) {
    constexpr int segments = 48;
    AppendCircleLines(vertices, transform, radius, segments, 0);
    AppendCircleLines(vertices, transform, radius, segments, 1);
    AppendCircleLines(vertices, transform, radius, segments, 2);
}

void AppendCylinderLines(std::vector<float>& vertices, const Affine3& transform, RealType radius, RealType height) {
    constexpr int segments = 48;
    const RealType half_height = height * static_cast<RealType>(0.5);

    for (int i = 0; i < segments; ++i) {
        const RealType a = static_cast<RealType>(2.0 * Math_PI * i / segments);
        const RealType b = static_cast<RealType>(2.0 * Math_PI * ((i + 1) % segments) / segments);
        const Vector3 top_from{std::cos(a) * radius, std::sin(a) * radius, half_height};
        const Vector3 top_to{std::cos(b) * radius, std::sin(b) * radius, half_height};
        const Vector3 bottom_from{std::cos(a) * radius, std::sin(a) * radius, -half_height};
        const Vector3 bottom_to{std::cos(b) * radius, std::sin(b) * radius, -half_height};

        AppendLine(vertices, transform, top_from, top_to);
        AppendLine(vertices, transform, bottom_from, bottom_to);

        if (i % 12 == 0) {
            AppendLine(vertices, transform, bottom_from, top_from);
        }
    }
}

void CollectCollisionLines(const Node* node, std::vector<float>& vertices) {
    const auto* collision_shape = Object::PointerCastTo<CollisionShape3D>(node);
    if (collision_shape && collision_shape->IsInsideTree() && collision_shape->IsVisibleInTree() &&
        !collision_shape->IsDisabled()) {
        const Ref<Shape3D>& shape = collision_shape->GetShape();
        const Affine3 transform = collision_shape->GetGlobalTransform();

        if (Ref<BoxShape3D> box = dynamic_pointer_cast<BoxShape3D>(shape); box.IsValid()) {
            AppendBoxLines(vertices, transform, box->GetSize());
        } else if (Ref<SphereShape3D> sphere = dynamic_pointer_cast<SphereShape3D>(shape); sphere.IsValid()) {
            AppendSphereLines(vertices, transform, static_cast<RealType>(sphere->GetRadius()));
        } else if (Ref<CylinderShape3D> cylinder = dynamic_pointer_cast<CylinderShape3D>(shape); cylinder.IsValid()) {
            AppendCylinderLines(vertices,
                                transform,
                                static_cast<RealType>(cylinder->GetRadius()),
                                static_cast<RealType>(cylinder->GetHeight()));
        }
    }

    for (std::size_t i = 0; i < node->GetChildCount(); ++i) {
        CollectCollisionLines(node->GetChild(static_cast<int>(i)), vertices);
    }
}

}

GLRendererDebugDraw::~GLRendererDebugDraw() {
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
    FreeLineBuffer(editor_grid_);
    FreeLineBuffer(world_axes_);
    FreeLineBuffer(collision_lines_);
}

void GLRendererDebugDraw::EnsureProgram() {
    if (program_ != 0) {
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

    GLuint vs = CompileDebugShader(GL_VERTEX_SHADER, vertex_shader);
    GLuint fs = CompileDebugShader(GL_FRAGMENT_SHADER, fragment_shader);
    if (vs == 0 || fs == 0) {
        return;
    }

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint status = GL_FALSE;
    glGetProgramiv(program_, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLsizei log_length = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &log_length);
        std::string log(static_cast<std::size_t>(std::max(log_length, GLsizei{1})), '\0');
        glGetProgramInfoLog(program_, log_length, &log_length, log.data());
        LOG_ERROR("Editor debug program link failed: {}", log);
        glDeleteProgram(program_);
        program_ = 0;
    }
}

void GLRendererDebugDraw::EnsureEditorGrid() {
    if (editor_grid_.vao != 0) {
        return;
    }

    constexpr int extent = 20;
    std::vector<float> vertices;
    vertices.reserve(static_cast<std::size_t>((extent * 2 + 1) * 4 * 3));

    for (int i = -extent; i <= extent; ++i) {
        const float p = static_cast<float>(i);
        const float e = static_cast<float>(extent);

        vertices.insert(vertices.end(), {-e, p, 0.0f, e, p, 0.0f});
        vertices.insert(vertices.end(), {p, -e, 0.0f, p, e, 0.0f});
    }

    glCreateVertexArrays(1, &editor_grid_.vao);
    glCreateBuffers(1, &editor_grid_.vertex_buffer);
    glNamedBufferData(editor_grid_.vertex_buffer,
                      static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                      vertices.data(),
                      GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(editor_grid_.vao, 0, editor_grid_.vertex_buffer, 0, 3 * sizeof(float));
    glEnableVertexArrayAttrib(editor_grid_.vao, 0);
    glVertexArrayAttribFormat(editor_grid_.vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(editor_grid_.vao, 0, 0);

    editor_grid_.vertex_count = static_cast<GLsizei>(vertices.size() / 3);
}

void GLRendererDebugDraw::EnsureWorldAxes() {
    if (world_axes_.vao != 0) {
        return;
    }

    constexpr float axis_length = 3.0f;
    const std::array<float, 18> vertices = {
            0.0f, 0.0f, 0.0f, axis_length, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, axis_length, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, axis_length,
    };

    glCreateVertexArrays(1, &world_axes_.vao);
    glCreateBuffers(1, &world_axes_.vertex_buffer);
    glNamedBufferData(world_axes_.vertex_buffer,
                      static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                      vertices.data(),
                      GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(world_axes_.vao, 0, world_axes_.vertex_buffer, 0, 3 * sizeof(float));
    glEnableVertexArrayAttrib(world_axes_.vao, 0);
    glVertexArrayAttribFormat(world_axes_.vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(world_axes_.vao, 0, 0);

    world_axes_.vertex_count = static_cast<GLsizei>(vertices.size() / 3);
}

void GLRendererDebugDraw::DrawEditorGrid() {
    EnsureEditorGrid();
    if (editor_grid_.vao == 0 || editor_grid_.vertex_count == 0) {
        return;
    }

    const Matrix4 model = Matrix4::Identity();
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_model"), 1, GL_FALSE, model.data());
    glUniform4f(glGetUniformLocation(program_, "u_color"), 0.32f, 0.34f, 0.38f, 1.0f);
    glBindVertexArray(editor_grid_.vao);
    glDrawArrays(GL_LINES, 0, editor_grid_.vertex_count);
}

void GLRendererDebugDraw::DrawWorldAxes() {
    EnsureWorldAxes();
    if (world_axes_.vao == 0 || world_axes_.vertex_count == 0) {
        return;
    }

    const Matrix4 model = Matrix4::Identity();
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_model"), 1, GL_FALSE, model.data());

    glBindVertexArray(world_axes_.vao);
    glLineWidth(2.0f);

    glUniform4f(glGetUniformLocation(program_, "u_color"), 0.95f, 0.18f, 0.18f, 1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glUniform4f(glGetUniformLocation(program_, "u_color"), 0.20f, 0.85f, 0.28f, 1.0f);
    glDrawArrays(GL_LINES, 2, 2);
    glUniform4f(glGetUniformLocation(program_, "u_color"), 0.22f, 0.44f, 1.0f, 1.0f);
    glDrawArrays(GL_LINES, 4, 2);

    glLineWidth(1.0f);
}

void GLRendererDebugDraw::DrawCollisionDebug(const Node* scene_root) {
    if (scene_root == nullptr) {
        return;
    }

    std::vector<float> vertices;
    CollectCollisionLines(scene_root, vertices);
    if (vertices.empty()) {
        collision_lines_.vertex_count = 0;
        return;
    }

    if (collision_lines_.vao == 0) {
        glCreateVertexArrays(1, &collision_lines_.vao);
        glCreateBuffers(1, &collision_lines_.vertex_buffer);
        glVertexArrayVertexBuffer(collision_lines_.vao, 0, collision_lines_.vertex_buffer, 0, 3 * sizeof(float));
        glEnableVertexArrayAttrib(collision_lines_.vao, 0);
        glVertexArrayAttribFormat(collision_lines_.vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribBinding(collision_lines_.vao, 0, 0);
    }

    glNamedBufferData(collision_lines_.vertex_buffer,
                      static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                      vertices.data(),
                      GL_STREAM_DRAW);
    collision_lines_.vertex_count = static_cast<GLsizei>(vertices.size() / 3);

    const Matrix4 model = Matrix4::Identity();
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_model"), 1, GL_FALSE, model.data());
    glUniform4f(glGetUniformLocation(program_, "u_color"), 0.15f, 0.95f, 0.72f, 0.85f);
    glBindVertexArray(collision_lines_.vao);
    glLineWidth(1.5f);
    glDrawArrays(GL_LINES, 0, collision_lines_.vertex_count);
    glLineWidth(1.0f);
}

void GLRendererDebugDraw::RenderEditorDebug(const RID& render_target, const Camera3D* camera, const Node* scene_root) {
    ERR_FAIL_COND(camera == nullptr);

    auto* rt = TextureStorage::GetInstance()->GetRenderTarget(render_target);
    ERR_FAIL_COND(rt == nullptr);
    if (rt->fbo == 0 || rt->size.x() <= 0 || rt->size.y() <= 0) {
        return;
    }

    EnsureProgram();
    if (program_ == 0) {
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);
    glViewport(0, 0, rt->size.x(), rt->size.y());
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(program_);
    const Matrix4 view = camera->GetViewMatrix();
    const Matrix4 projection = camera->GetProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_projection"), 1, GL_FALSE, projection.data());

    DrawEditorGrid();
    DrawWorldAxes();
    DrawCollisionDebug(scene_root);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}
