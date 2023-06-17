/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-21
*/

#include "gobot/rendering/mesh_storage.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

MeshStorage *MeshStorage::s_singleton = nullptr;

MeshStorage::MeshStorage() {
    s_singleton = this;
}

MeshStorage::~MeshStorage() {
    s_singleton = nullptr;
}

MeshStorage* MeshStorage::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize MeshStorage");
    return s_singleton;
}

void MeshStorage::MeshInitialize(const RID& mesh_rid) {
    mesh_owner_.InitializeRID(mesh_rid, Mesh());
}

void MeshStorage::MeshClear(const RID& mesh_rid) {
    Mesh* mesh = mesh_owner_.GetOrNull(mesh_rid);
    ERR_FAIL_COND(!mesh);
}

void MeshStorage::MeshFree(const RID& rid) {
    Mesh* mesh = mesh_owner_.GetOrNull(rid);
    ERR_FAIL_COND(mesh == nullptr);
    mesh_owner_.Free(rid);
}

}
