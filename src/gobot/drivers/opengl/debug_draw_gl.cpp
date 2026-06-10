/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/drivers/opengl/debug_draw_gl.hpp"

#include "gobot/drivers/opengl/texture_storage.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"
#include "gobot/physics/backends/null_physics_world.hpp"
#include "gobot/physics/physics_types.hpp"
#include "gobot/physics/physics_world.hpp"
#include "gobot/rendering/scene_render_items.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/resources/box_shape_3d.hpp"
#include "gobot/scene/resources/capsule_shape_3d.hpp"
#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/sphere_shape_3d.hpp"
#include "glsl_shader_hpp/debug_draw_frag.hpp"
#include "glsl_shader_hpp/debug_draw_vert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <optional>
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

void AppendWorldLine(std::vector<float>& vertices, const Vector3& from, const Vector3& to) {
    PushWorldVertex(vertices, from);
    PushWorldVertex(vertices, to);
}

void AppendCross(std::vector<float>& vertices, const Vector3& center, RealType radius) {
    AppendWorldLine(vertices, center - Vector3::UnitX() * radius, center + Vector3::UnitX() * radius);
    AppendWorldLine(vertices, center - Vector3::UnitY() * radius, center + Vector3::UnitY() * radius);
    AppendWorldLine(vertices, center - Vector3::UnitZ() * radius, center + Vector3::UnitZ() * radius);
}

Vector3 SpherePoint(const Vector3& center, RealType radius, RealType theta, RealType phi) {
    const RealType sin_theta = std::sin(theta);
    return center + Vector3{
            std::cos(phi) * sin_theta * radius,
            std::cos(theta) * radius,
            std::sin(phi) * sin_theta * radius,
    };
}

void AppendSphereTriangles(std::vector<float>& vertices, const Vector3& center, RealType radius) {
    constexpr int stack_count = 12;
    constexpr int sector_count = 24;
    for (int stack = 0; stack < stack_count; ++stack) {
        const RealType theta0 = static_cast<RealType>(Math_PI * stack / stack_count);
        const RealType theta1 = static_cast<RealType>(Math_PI * (stack + 1) / stack_count);
        for (int segment = 0; segment < sector_count; ++segment) {
            const RealType phi0 = static_cast<RealType>(2.0 * Math_PI * segment / sector_count);
            const RealType phi1 = static_cast<RealType>(2.0 * Math_PI * (segment + 1) / sector_count);
            const Vector3 p00 = SpherePoint(center, radius, theta0, phi0);
            const Vector3 p10 = SpherePoint(center, radius, theta1, phi0);
            const Vector3 p11 = SpherePoint(center, radius, theta1, phi1);
            const Vector3 p01 = SpherePoint(center, radius, theta0, phi1);

            if (stack != 0) {
                PushWorldVertex(vertices, p00);
                PushWorldVertex(vertices, p10);
                PushWorldVertex(vertices, p01);
            }
            if (stack != stack_count - 1) {
                PushWorldVertex(vertices, p01);
                PushWorldVertex(vertices, p10);
                PushWorldVertex(vertices, p11);
            }
        }
    }
}

void AppendArrow(std::vector<float>& vertices, const Vector3& from, const Vector3& to) {
    AppendWorldLine(vertices, from, to);

    const Vector3 delta = to - from;
    const RealType length = delta.norm();
    if (length <= CMP_EPSILON) {
        return;
    }

    const Vector3 direction = delta / length;
    Vector3 side = direction.cross(Vector3::UnitZ());
    if (side.norm() <= CMP_EPSILON) {
        side = direction.cross(Vector3::UnitY());
    }
    if (side.norm() <= CMP_EPSILON) {
        return;
    }
    side.normalize();

    const RealType head_length = std::min<RealType>(0.06, length * static_cast<RealType>(0.35));
    const RealType head_width = head_length * static_cast<RealType>(0.45);
    const Vector3 base = to - direction * head_length;
    AppendWorldLine(vertices, to, base + side * head_width);
    AppendWorldLine(vertices, to, base - side * head_width);
}

struct ArrowBatch {
    Color color;
    std::vector<float> vertices;
};

bool SameColor(const Color& left, const Color& right) {
    return std::abs(left.red() - right.red()) <= 1.0e-4f &&
           std::abs(left.green() - right.green()) <= 1.0e-4f &&
           std::abs(left.blue() - right.blue()) <= 1.0e-4f &&
           std::abs(left.alpha() - right.alpha()) <= 1.0e-4f;
}

std::vector<ArrowBatch> BuildArrowBatches(const std::vector<DebugArrow>& arrows) {
    std::vector<ArrowBatch> batches;
    for (const DebugArrow& arrow : arrows) {
        const Vector3 scaled_vector = arrow.vector * arrow.scale;
        if (scaled_vector.squaredNorm() <= CMP_EPSILON2) {
            continue;
        }

        auto iter = std::ranges::find_if(batches, [&](const ArrowBatch& batch) {
            return SameColor(batch.color, arrow.color);
        });
        if (iter == batches.end()) {
            batches.push_back({arrow.color, {}});
            iter = std::prev(batches.end());
        }
        AppendArrow(iter->vertices, arrow.start, arrow.start + scaled_vector);
    }
    return batches;
}

void EnsureLineBuffer(GLRendererDebugDraw::LineBuffer& buffer) {
    if (buffer.vao != 0) {
        return;
    }

    glCreateVertexArrays(1, &buffer.vao);
    glCreateBuffers(1, &buffer.vertex_buffer);
    glVertexArrayVertexBuffer(buffer.vao, 0, buffer.vertex_buffer, 0, 3 * sizeof(float));
    glEnableVertexArrayAttrib(buffer.vao, 0);
    glVertexArrayAttribFormat(buffer.vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(buffer.vao, 0, 0);
}

void DrawLineBuffer(GLRendererDebugDraw::LineBuffer& buffer,
                    const std::vector<float>& vertices,
                    GLuint program,
                    float red,
                    float green,
                    float blue,
                    float alpha,
                    float line_width) {
    if (vertices.empty()) {
        buffer.vertex_count = 0;
        return;
    }

    EnsureLineBuffer(buffer);
    glNamedBufferData(buffer.vertex_buffer,
                      static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                      vertices.data(),
                      GL_STREAM_DRAW);
    buffer.vertex_count = static_cast<GLsizei>(vertices.size() / 3);

    const Matrix4 model = Matrix4::Identity();
    glUniformMatrix4fv(glGetUniformLocation(program, "u_model"), 1, GL_FALSE, model.data());
    glUniform4f(glGetUniformLocation(program, "u_color"), red, green, blue, alpha);
    glBindVertexArray(buffer.vao);
    glLineWidth(line_width);
    glDrawArrays(GL_LINES, 0, buffer.vertex_count);
    glLineWidth(1.0f);
}

void DrawTriangleBuffer(GLRendererDebugDraw::LineBuffer& buffer,
                        const std::vector<float>& vertices,
                        GLuint program,
                        float red,
                        float green,
                        float blue,
                        float alpha) {
    if (vertices.empty()) {
        buffer.vertex_count = 0;
        return;
    }

    EnsureLineBuffer(buffer);
    glNamedBufferData(buffer.vertex_buffer,
                      static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                      vertices.data(),
                      GL_STREAM_DRAW);
    buffer.vertex_count = static_cast<GLsizei>(vertices.size() / 3);

    const Matrix4 model = Matrix4::Identity();
    glUniformMatrix4fv(glGetUniformLocation(program, "u_model"), 1, GL_FALSE, model.data());
    glUniform4f(glGetUniformLocation(program, "u_color"), red, green, blue, alpha);
    glBindVertexArray(buffer.vao);
    glDrawArrays(GL_TRIANGLES, 0, buffer.vertex_count);
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

void AppendCapsuleLines(std::vector<float>& vertices, const Affine3& transform, RealType radius, RealType height) {
    constexpr int segments = 48;
    const RealType half_height = height * static_cast<RealType>(0.5);

    AppendCylinderLines(vertices, transform, radius, height);

    Affine3 top_transform = transform;
    top_transform.translation() = transform * Vector3{0.0, 0.0, half_height};
    AppendSphereLines(vertices, top_transform, radius);

    Affine3 bottom_transform = transform;
    bottom_transform.translation() = transform * Vector3{0.0, 0.0, -half_height};
    AppendSphereLines(vertices, bottom_transform, radius);
}

void CollectCollisionLines(const SceneRenderItems& render_items, std::vector<float>& vertices) {
    for (const CollisionDebugRenderItem& item : render_items.collision_shapes) {
        if (Ref<BoxShape3D> box = dynamic_pointer_cast<BoxShape3D>(item.shape); box.IsValid()) {
            AppendBoxLines(vertices, item.transform, box->GetSize());
        } else if (Ref<SphereShape3D> sphere = dynamic_pointer_cast<SphereShape3D>(item.shape); sphere.IsValid()) {
            AppendSphereLines(vertices, item.transform, static_cast<RealType>(sphere->GetRadius()));
        } else if (Ref<CylinderShape3D> cylinder = dynamic_pointer_cast<CylinderShape3D>(item.shape); cylinder.IsValid()) {
            AppendCylinderLines(vertices,
                                item.transform,
                                static_cast<RealType>(cylinder->GetRadius()),
                                static_cast<RealType>(cylinder->GetHeight()));
        } else if (Ref<CapsuleShape3D> capsule = dynamic_pointer_cast<CapsuleShape3D>(item.shape); capsule.IsValid()) {
            AppendCapsuleLines(vertices,
                               item.transform,
                               static_cast<RealType>(capsule->GetRadius()),
                               static_cast<RealType>(capsule->GetHeight()));
        }
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
    FreeLineBuffer(height_scanner_ray_lines_);
    FreeLineBuffer(height_scanner_hit_spheres_);
    FreeLineBuffer(height_scanner_normal_lines_);
    FreeLineBuffer(contact_point_lines_);
    FreeLineBuffer(contact_normal_lines_);
    FreeLineBuffer(contact_force_lines_);
    FreeLineBuffer(debug_arrow_lines_);
}

void GLRendererDebugDraw::EnsureProgram() {
    if (program_ != 0) {
        return;
    }

    GLuint vs = CompileDebugShader(GL_VERTEX_SHADER, DEBUG_DRAW_VERT);
    GLuint fs = CompileDebugShader(GL_FRAGMENT_SHADER, DEBUG_DRAW_FRAG);
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

void GLRendererDebugDraw::DrawCollisionDebug(const SceneRenderItems& render_items) {
    std::vector<float> vertices;
    CollectCollisionLines(render_items, vertices);
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

void GLRendererDebugDraw::DrawHeightScannerDebug(const PhysicsSceneState* physics_state) {
    std::vector<float> ray_vertices;
    std::vector<float> hit_vertices;
    std::vector<float> normal_vertices;
    auto append_sensor = [&](const PhysicsSensorState& sensor) {
        if ((sensor.type != PhysicsSensorType::RayCast &&
             sensor.type != PhysicsSensorType::TerrainHeight &&
             sensor.type != PhysicsSensorType::HeightScanner) ||
            !sensor.enabled ||
            !sensor.visualize_debug) {
            return;
        }
        for (const PhysicsSensorRaycastHit& hit : sensor.hits) {
            const Vector3 delta = hit.point - hit.origin;
            const RealType length = delta.norm();
            if (length > CMP_EPSILON) {
                const Vector3 direction = delta / length;
                const RealType guide_offset = std::min<RealType>(0.18, length * static_cast<RealType>(0.18));
                AppendWorldLine(ray_vertices, hit.origin + direction * guide_offset, hit.point);
            }

            if (!hit.hit) {
                continue;
            }

            const RealType normal_length = hit.normal.norm();
            constexpr RealType kHitSphereRadius = 0.022;
            AppendSphereTriangles(hit_vertices, hit.point, kHitSphereRadius);

            if (normal_length > CMP_EPSILON) {
                constexpr RealType kNormalVisualLength = 0.16;
                AppendArrow(normal_vertices,
                            hit.point,
                            hit.point + hit.normal / normal_length * kNormalVisualLength);
            }
        }
    };
    if (physics_state != nullptr) {
        for (const PhysicsRobotState& robot : physics_state->robots) {
            for (const PhysicsSensorState& sensor : robot.sensors) {
                append_sensor(sensor);
            }
        }
        for (const PhysicsSensorState& sensor : physics_state->loose_sensors) {
            append_sensor(sensor);
        }
    }

    DrawLineBuffer(height_scanner_ray_lines_, ray_vertices, program_, 0.35f, 0.95f, 0.22f, 0.42f, 1.0f);
    DrawTriangleBuffer(height_scanner_hit_spheres_, hit_vertices, program_, 0.0f, 0.93f, 1.0f, 0.98f);
    DrawLineBuffer(height_scanner_normal_lines_, normal_vertices, program_, 1.0f, 0.88f, 0.05f, 0.95f, 1.75f);
}

void GLRendererDebugDraw::DrawContactDebug(const PhysicsWorld* physics_world) {
    if (physics_world == nullptr) {
        contact_point_lines_.vertex_count = 0;
        contact_normal_lines_.vertex_count = 0;
        contact_force_lines_.vertex_count = 0;
        return;
    }

    const PhysicsSceneState& physics_state = physics_world->GetSceneState();
    const PhysicsWorldSettings& settings = physics_world->GetSettings();
    std::vector<std::pair<std::string, std::string>> visualized_sensor_links;
    if (!settings.debug_draw_contacts) {
        for (const PhysicsRobotState& robot : physics_state.robots) {
            for (const PhysicsSensorState& sensor : robot.sensors) {
                if (sensor.type == PhysicsSensorType::Contact && sensor.enabled && sensor.visualize_debug) {
                    visualized_sensor_links.emplace_back(robot.name, sensor.link_name);
                }
            }
        }
        if (visualized_sensor_links.empty()) {
            contact_point_lines_.vertex_count = 0;
            contact_normal_lines_.vertex_count = 0;
            contact_force_lines_.vertex_count = 0;
            return;
        }
    }

    auto should_draw_contact = [&](const PhysicsContactState& contact) {
        if (settings.debug_draw_contacts) {
            return true;
        }
        return std::any_of(visualized_sensor_links.begin(),
                           visualized_sensor_links.end(),
                           [&](const auto& link_key) {
                               return link_key.first == contact.robot_name &&
                                      link_key.second == contact.link_name;
                           });
    };

    std::vector<float> point_vertices;
    std::vector<float> normal_vertices;
    std::vector<float> force_vertices;
    for (const PhysicsContactState& contact : physics_state.contacts) {
        if (!should_draw_contact(contact)) {
            continue;
        }

        AppendCross(point_vertices, contact.position, 0.035);

        const RealType normal_norm = contact.normal.norm();
        if (normal_norm > CMP_EPSILON) {
            AppendArrow(normal_vertices,
                        contact.position,
                        contact.position + contact.normal / normal_norm * static_cast<RealType>(0.14));
        }

        const RealType force_norm = contact.force.norm();
        if (settings.debug_draw_contact_forces && force_norm > CMP_EPSILON) {
            const RealType arrow_length =
                    std::min(settings.debug_contact_force_max_length,
                             settings.debug_contact_force_scale * std::log1p(force_norm));
            if (arrow_length > CMP_EPSILON) {
                AppendArrow(force_vertices,
                            contact.position,
                            contact.position + contact.force / force_norm * arrow_length);
            }
        }
    }

    DrawLineBuffer(contact_point_lines_, point_vertices, program_, 1.0f, 0.28f, 0.18f, 0.9f, 2.0f);
    DrawLineBuffer(contact_normal_lines_, normal_vertices, program_, 0.20f, 0.70f, 1.0f, 0.9f, 2.0f);
    DrawLineBuffer(contact_force_lines_, force_vertices, program_, 1.0f, 0.05f, 0.02f, 0.95f, 3.0f);
}

void GLRendererDebugDraw::RenderEditorDebug(const RID& render_target,
                                            const Camera3D* camera,
                                            const Node* scene_root,
                                            const PhysicsWorld* physics_world) {
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

    const SceneRenderItems render_items = CollectSceneRenderItems(scene_root);
    std::optional<NullPhysicsWorld> preview_world;
    if (physics_world == nullptr && scene_root != nullptr) {
        preview_world.emplace();
        if (!preview_world->BuildFromScene(scene_root)) {
            preview_world.reset();
        }
    }
    const PhysicsSceneState* physics_state = physics_world != nullptr
                                                     ? &physics_world->GetSceneState()
                                                     : (preview_world.has_value() ? &preview_world->GetSceneState() : nullptr);
    DrawEditorGrid();
    DrawWorldAxes();
    DrawCollisionDebug(render_items);
    DrawHeightScannerDebug(physics_state);

    glDisable(GL_DEPTH_TEST);
    DrawContactDebug(physics_world);
    glEnable(GL_DEPTH_TEST);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLRendererDebugDraw::RenderDebugArrows(const RID& render_target,
                                            const Camera3D* camera,
                                            const std::vector<DebugArrow>& arrows) {
    ERR_FAIL_COND(camera == nullptr);
    if (arrows.empty()) {
        debug_arrow_lines_.vertex_count = 0;
        return;
    }

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

    for (const ArrowBatch& batch : BuildArrowBatches(arrows)) {
        DrawLineBuffer(debug_arrow_lines_,
                       batch.vertices,
                       program_,
                       batch.color.red(),
                       batch.color.green(),
                       batch.color.blue(),
                       batch.color.alpha(),
                       4.0f);
    }

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}
