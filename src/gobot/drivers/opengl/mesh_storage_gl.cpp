/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-8-9.
*/

#include "gobot/drivers/opengl/mesh_storage_gl.hpp"

#include "gobot/error_macros.hpp"
#include "gobot/core/math/math_defs.hpp"

#include <array>
#include <algorithm>
#include <cmath>

namespace gobot::opengl {

namespace {

void PushVertex(std::vector<float>& vertices, RealType x, RealType y, RealType z) {
    vertices.push_back(static_cast<float>(x));
    vertices.push_back(static_cast<float>(y));
    vertices.push_back(static_cast<float>(z));
}

Vector3 ReadVertex(const std::vector<float>& vertices, uint32_t index) {
    const std::size_t offset = static_cast<std::size_t>(index) * 3;
    return {vertices[offset], vertices[offset + 1], vertices[offset + 2]};
}

std::vector<float> GenerateSmoothNormals(const std::vector<float>& vertices, const std::vector<uint32_t>& indices) {
    std::vector<Vector3> normals(vertices.size() / 3, Vector3::Zero());

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t ia = indices[i];
        const uint32_t ib = indices[i + 1];
        const uint32_t ic = indices[i + 2];
        if (static_cast<std::size_t>(std::max({ia, ib, ic})) >= normals.size()) {
            continue;
        }

        const Vector3 a = ReadVertex(vertices, ia);
        const Vector3 b = ReadVertex(vertices, ib);
        const Vector3 c = ReadVertex(vertices, ic);
        Vector3 normal = (b - a).cross(c - a);
        const RealType length = normal.norm();
        if (length <= CMP_EPSILON) {
            continue;
        }

        normal /= length;
        normals[ia] += normal;
        normals[ib] += normal;
        normals[ic] += normal;
    }

    std::vector<float> packed_normals;
    packed_normals.reserve(vertices.size());
    for (Vector3 normal : normals) {
        const RealType length = normal.norm();
        if (length <= CMP_EPSILON) {
            normal = Vector3::UnitZ();
        } else {
            normal /= length;
        }
        PushVertex(packed_normals, normal.x(), normal.y(), normal.z());
    }

    return packed_normals;
}

void SetMeshData(GLMeshData* mesh,
                 std::vector<float> vertices,
                 std::vector<uint32_t> indices,
                 std::vector<float> normals = {}) {
    mesh->vertices = std::move(vertices);
    mesh->normals = normals.size() == mesh->vertices.size()
            ? std::move(normals)
            : GenerateSmoothNormals(mesh->vertices, indices);
    mesh->indices = std::move(indices);
    mesh->index_count = static_cast<GLsizei>(mesh->indices.size());
    mesh->dirty = true;
}

} // namespace

GLMeshStorage* GLMeshStorage::s_singleton = nullptr;

MeshStorage* GLMeshStorage::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize MeshStorage");
    return s_singleton;
}

GLMeshStorage::GLMeshStorage() {
    s_singleton = this;
}

GLMeshStorage::~GLMeshStorage() {
    s_singleton = nullptr;
}


RID GLMeshStorage::MeshAllocate() {
    return mesh_owner_.AllocateRID();
}

void GLMeshStorage::MeshInitialize(const RID& p_rid)  {
    mesh_owner_.InitializeRID(p_rid, GLMeshData());
}

void GLMeshStorage::MeshSetBox(const RID& p_rid, const Vector3& size) {
    GLMeshData* mesh = mesh_owner_.GetOrNull(p_rid);
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

    struct BoxFace {
        std::array<uint32_t, 4> corners;
        Vector3 normal;
    };

    const std::array<BoxFace, 6> faces = {
            BoxFace{{0, 1, 2, 3}, Vector3{0.0f, 0.0f, -1.0f}},
            BoxFace{{4, 5, 6, 7}, Vector3{0.0f, 0.0f, 1.0f}},
            BoxFace{{0, 1, 5, 4}, Vector3{0.0f, -1.0f, 0.0f}},
            BoxFace{{3, 2, 6, 7}, Vector3{0.0f, 1.0f, 0.0f}},
            BoxFace{{1, 2, 6, 5}, Vector3{1.0f, 0.0f, 0.0f}},
            BoxFace{{0, 3, 7, 4}, Vector3{-1.0f, 0.0f, 0.0f}},
    };

    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<uint32_t> indices;
    vertices.reserve(faces.size() * 4 * 3);
    normals.reserve(faces.size() * 4 * 3);
    indices.reserve(faces.size() * 6);
    for (const BoxFace& face : faces) {
        const uint32_t base = static_cast<uint32_t>(vertices.size() / 3);
        for (uint32_t corner : face.corners) {
            const Vector3& point = p[corner];
            PushVertex(vertices, point.x(), point.y(), point.z());
            PushVertex(normals, face.normal.x(), face.normal.y(), face.normal.z());
        }
        indices.insert(indices.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
    }
    SetMeshData(mesh, std::move(vertices), std::move(indices), std::move(normals));
}

void GLMeshStorage::MeshSetSurface(const RID& p_rid,
                                   const std::vector<Vector3>& surface_vertices,
                                   const std::vector<uint32_t>& surface_indices,
                                   const std::vector<Vector3>& surface_normals) {
    GLMeshData* mesh = mesh_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(mesh == nullptr);

    std::vector<float> vertices;
    vertices.reserve(surface_vertices.size() * 3);
    for (const Vector3& vertex : surface_vertices) {
        PushVertex(vertices, vertex.x(), vertex.y(), vertex.z());
    }

    std::vector<float> normals;
    normals.reserve(surface_normals.size() * 3);
    for (const Vector3& normal : surface_normals) {
        PushVertex(normals, normal.x(), normal.y(), normal.z());
    }

    SetMeshData(mesh, std::move(vertices), surface_indices, std::move(normals));
}

void GLMeshStorage::MeshSetCylinder(const RID& p_rid, RealType radius, RealType height, int radial_segments) {
    GLMeshData* mesh = mesh_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(mesh == nullptr);

    radius = std::max(radius, static_cast<RealType>(0.0));
    height = std::max(height, static_cast<RealType>(0.0));
    radial_segments = std::max(radial_segments, 3);

    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(static_cast<std::size_t>(2 + radial_segments * 2) * 3);
    indices.reserve(static_cast<std::size_t>(radial_segments) * 12);

    const RealType half_height = height * static_cast<RealType>(0.5);
    PushVertex(vertices, 0.0, half_height, 0.0);
    PushVertex(vertices, 0.0, -half_height, 0.0);

    for (int i = 0; i < radial_segments; ++i) {
        const RealType angle = static_cast<RealType>(2.0 * Math_PI * i / radial_segments);
        const RealType x = std::cos(angle) * radius;
        const RealType z = std::sin(angle) * radius;
        PushVertex(vertices, x, half_height, z);
        PushVertex(vertices, x, -half_height, z);
    }

    constexpr uint32_t top_center = 0;
    constexpr uint32_t bottom_center = 1;
    for (int i = 0; i < radial_segments; ++i) {
        const int next = (i + 1) % radial_segments;
        const uint32_t top = static_cast<uint32_t>(2 + i * 2);
        const uint32_t bottom = top + 1;
        const uint32_t next_top = static_cast<uint32_t>(2 + next * 2);
        const uint32_t next_bottom = next_top + 1;

        indices.insert(indices.end(), {top_center, next_top, top});
        indices.insert(indices.end(), {bottom_center, bottom, next_bottom});
        indices.insert(indices.end(), {top, next_top, next_bottom});
        indices.insert(indices.end(), {top, next_bottom, bottom});
    }

    SetMeshData(mesh, std::move(vertices), std::move(indices));
}

void GLMeshStorage::MeshSetSphere(const RID& p_rid, RealType radius, int radial_segments, int rings) {
    GLMeshData* mesh = mesh_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(mesh == nullptr);

    radius = std::max(radius, static_cast<RealType>(0.0));
    radial_segments = std::max(radial_segments, 3);
    rings = std::max(rings, 2);

    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(static_cast<std::size_t>(2 + (rings - 1) * radial_segments) * 3);
    indices.reserve(static_cast<std::size_t>(radial_segments * rings) * 6);

    const uint32_t top_index = 0;
    PushVertex(vertices, 0.0, radius, 0.0);

    for (int ring = 1; ring < rings; ++ring) {
        const RealType theta = static_cast<RealType>(Math_PI * ring / rings);
        const RealType y = std::cos(theta) * radius;
        const RealType ring_radius = std::sin(theta) * radius;
        for (int segment = 0; segment < radial_segments; ++segment) {
            const RealType phi = static_cast<RealType>(2.0 * Math_PI * segment / radial_segments);
            PushVertex(vertices, std::cos(phi) * ring_radius, y, std::sin(phi) * ring_radius);
        }
    }

    const uint32_t bottom_index = static_cast<uint32_t>(vertices.size() / 3);
    PushVertex(vertices, 0.0, -radius, 0.0);

    auto ring_vertex = [radial_segments](int ring, int segment) -> uint32_t {
        return static_cast<uint32_t>(1 + (ring - 1) * radial_segments + (segment % radial_segments));
    };

    for (int segment = 0; segment < radial_segments; ++segment) {
        const uint32_t current = ring_vertex(1, segment);
        const uint32_t next = ring_vertex(1, segment + 1);
        indices.insert(indices.end(), {top_index, current, next});
    }

    for (int ring = 1; ring < rings - 1; ++ring) {
        for (int segment = 0; segment < radial_segments; ++segment) {
            const uint32_t a = ring_vertex(ring, segment);
            const uint32_t b = ring_vertex(ring, segment + 1);
            const uint32_t c = ring_vertex(ring + 1, segment);
            const uint32_t d = ring_vertex(ring + 1, segment + 1);
            indices.insert(indices.end(), {a, c, b});
            indices.insert(indices.end(), {b, c, d});
        }
    }

    for (int segment = 0; segment < radial_segments; ++segment) {
        const uint32_t current = ring_vertex(rings - 1, segment);
        const uint32_t next = ring_vertex(rings - 1, segment + 1);
        indices.insert(indices.end(), {bottom_index, next, current});
    }

    SetMeshData(mesh, std::move(vertices), std::move(indices));
}

bool GLMeshStorage::OwnsMesh(const RID& p_rid) const {
    return mesh_owner_.Owns(p_rid);
}

void GLMeshStorage::MeshFree(const RID& p_rid) {
    GLMeshData* mesh = mesh_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(mesh == nullptr);
    if (mesh->vao != 0) {
        glDeleteVertexArrays(1, &mesh->vao);
    }
    if (mesh->vertex_buffer != 0) {
        glDeleteBuffers(1, &mesh->vertex_buffer);
    }
    if (mesh->normal_buffer != 0) {
        glDeleteBuffers(1, &mesh->normal_buffer);
    }
    if (mesh->index_buffer != 0) {
        glDeleteBuffers(1, &mesh->index_buffer);
    }
    mesh_owner_.Free(p_rid);
}

}
