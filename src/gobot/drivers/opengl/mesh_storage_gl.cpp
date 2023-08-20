/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-8-9.
*/

#include "gobot/drivers/opengl/mesh_storage_gl.hpp"

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
    mesh_owner_.InitializeRID(p_rid, Mesh());
}

void GLMeshStorage::MeshFree(const RID& p_rid) {
    Mesh* mesh = mesh_owner_.GetOrNull(p_rid);
    ERR_FAIL_COND(mesh == nullptr);
    mesh_owner_.Free(p_rid);
}

}