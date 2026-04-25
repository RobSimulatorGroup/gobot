/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-8-9.
*/

#pragma once

#include "glad/glad.h"
#include "gobot/rendering/mesh_storage.hpp"

namespace gobot {

class Node;

}

namespace gobot::opengl {

class GLMeshStorage : public MeshStorage {
public:
    GLMeshStorage();

    ~GLMeshStorage() override;

    RID MeshAllocate() override;

    void MeshInitialize(const RID& p_rid) override;

    void MeshSetBox(const RID& p_rid, const Vector3& size) override;

    void RenderScene(const RID& render_target, const SceneTree* scene_tree, const Camera3D* camera) override;

    bool OwnsMesh(const RID& p_rid) const override;

    void MeshFree(const RID& p_rid) override;

public:
    static MeshStorage* GetInstance();

private:
    static GLMeshStorage* s_singleton;

    struct MeshData {
        std::vector<float> vertices;
        std::vector<uint32_t> indices;
        GLuint vao = 0;
        GLuint vertex_buffer = 0;
        GLuint index_buffer = 0;
        GLsizei index_count = 0;
        bool dirty = true;
    };

    mutable RID_Owner<MeshData> mesh_owner_;

    GLuint default_program_ = 0;

    void EnsureDefaultProgram();

    void UploadMesh(MeshData* mesh);

    void DrawNode(Node* node, const Matrix4& view, const Matrix4& projection);
};


}
