/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-8-9.
*/

#include "gobot/drivers/opengl/mesh_storage_gl.hpp"

#include "gobot/error_macros.hpp"

#include <array>

namespace gobot::opengl {

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
    GLMeshData* mesh = mesh_owner_.GetOrNull(p_rid);
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

}
