/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-21
*/

#pragma once

#include "gobot/rendering/render_rid_owner.hpp"

namespace gobot {

class MeshStorage {
public:
    MeshStorage();

    virtual ~MeshStorage();

    static MeshStorage* GetInstance();

    void MeshInitialize(const RenderRID& mesh_rid);

    void MeshClear(const RenderRID& mesh_rid);

    void MeshFree(const RenderRID& rid);

private:
    static MeshStorage* s_singleton;

    struct Mesh {
        RenderRID vertex_buffer{};
        RenderRID index_buffer{};
        RenderRID material{};
    };

    mutable RenderRID_Owner<Mesh> mesh_owner_;
};

}
