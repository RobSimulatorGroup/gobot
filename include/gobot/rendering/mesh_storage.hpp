/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-21
*/

#pragma once

#include "gobot/core/rid.hpp"
#include "gobot/core/rid_owner.hpp"

namespace gobot {

class MeshStorage {
public:
    MeshStorage();

    virtual ~MeshStorage();

    static MeshStorage* GetInstance();

    void MeshInitialize(const RID& mesh_rid);

    void MeshClear(const RID& mesh_rid);

    void MeshFree(const RID& rid);

private:
    static MeshStorage* s_singleton;

    struct Mesh {
        RID vertex_buffer{};
        RID index_buffer{};
        RID material{};
    };

    mutable RID_Owner<Mesh> mesh_owner_;
};

}
