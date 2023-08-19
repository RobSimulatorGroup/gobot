/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-8-9.
*/

#pragma once

#include "gobot/rendering/mesh_storage.hpp"

namespace gobot::opengl {

class GLMeshStorage : public MeshStorage {
public:
    GLMeshStorage();

    ~GLMeshStorage() override;

    RID MeshAllocate() override;

    void MeshInitialize(const RID& p_rid) override;

    void MeshFree(const RID& p_rid) override;

public:
    static MeshStorage* GetInstance();

private:
    static GLMeshStorage* s_singleton;

    struct Mesh {
        RID vertex_buffer{};
        RID index_buffer{};
        RID material{};
    };

    mutable RID_Owner<Mesh> mesh_owner_;
};


}