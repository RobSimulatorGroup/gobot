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

#include <algorithm>
#include <array>
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

}

GLRendererDebugDraw::~GLRendererDebugDraw() {
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
    FreeLineBuffer(editor_grid_);
    FreeLineBuffer(world_axes_);
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

void GLRendererDebugDraw::RenderEditorDebug(const RID& render_target, const Camera3D* camera) {
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

    glUseProgram(program_);
    const Matrix4 view = camera->GetViewMatrix();
    const Matrix4 projection = camera->GetProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_view"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_projection"), 1, GL_FALSE, projection.data());

    DrawEditorGrid();
    DrawWorldAxes();

    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}
