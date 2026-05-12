/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-8-9.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "glad/glad.h"
#include "gobot/rendering/mesh_storage.hpp"

#include <vector>

namespace gobot::opengl {

struct GLMeshData {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<uint32_t> indices;
    GLuint vao = 0;
    GLuint vertex_buffer = 0;
    GLuint normal_buffer = 0;
    GLuint index_buffer = 0;
    GLsizei index_count = 0;
    bool dirty = true;
};

class GLMeshStorage : public MeshStorage {
public:
    GLMeshStorage();

    ~GLMeshStorage() override;

    RID MeshAllocate() override;

    void MeshInitialize(const RID& p_rid) override;

    void MeshSetBox(const RID& p_rid, const Vector3& size) override;

    void MeshSetSurface(const RID& p_rid,
                        const std::vector<Vector3>& vertices,
                        const std::vector<uint32_t>& indices,
                        const std::vector<Vector3>& normals) override;

    void MeshSetCylinder(const RID& p_rid, RealType radius, RealType height, int radial_segments) override;

    void MeshSetSphere(const RID& p_rid, RealType radius, int radial_segments, int rings) override;

    bool OwnsMesh(const RID& p_rid) const override;

    void MeshFree(const RID& p_rid) override;

public:
    static MeshStorage* GetInstance();

private:
    static GLMeshStorage* s_singleton;

    mutable RID_Owner<GLMeshData> mesh_owner_;

    friend class GLRasterizerScene;
};


}
