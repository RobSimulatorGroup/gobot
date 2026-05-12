/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/array_mesh.hpp"

#include "gobot/core/object.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

ArrayMesh::ArrayMesh() {
    if (RenderServer::HasInstance()) {
        mesh_ = RenderServer::GetInstance()->MeshCreate();
    }
}

ArrayMesh::~ArrayMesh() {
    if (RenderServer::HasInstance()) {
        RS::GetInstance()->Free(mesh_);
    }
}

void ArrayMesh::SetSurface(std::vector<Vector3> vertices,
                           std::vector<uint32_t> indices,
                           std::vector<Vector3> normals) {
    vertices_ = std::move(vertices);
    indices_ = std::move(indices);
    normals_ = std::move(normals);
    if (normals_.size() != vertices_.size()) {
        normals_.clear();
    }
    UploadSurface();
}

const std::vector<Vector3>& ArrayMesh::GetVertices() const {
    return vertices_;
}

const std::vector<uint32_t>& ArrayMesh::GetIndices() const {
    return indices_;
}

const std::vector<Vector3>& ArrayMesh::GetNormals() const {
    return normals_;
}

void ArrayMesh::SetMaterial(const Ref<Material>& material) {
    material_ = material;
}

const Ref<Material>& ArrayMesh::GetMaterial() const {
    return material_;
}

RID ArrayMesh::GetRid() const {
    return mesh_;
}

void ArrayMesh::UploadSurface() {
    if (!RenderServer::HasInstance()) {
        return;
    }
    if (mesh_.IsNull()) {
        mesh_ = RenderServer::GetInstance()->MeshCreate();
    }
    RS::GetInstance()->MeshSetSurface(mesh_, vertices_, indices_, normals_);
}

} // namespace gobot

GOBOT_REGISTRATION {
    USING_ENUM_BITWISE_OPERATORS;

    Class_<gobot::ArrayMesh>("ArrayMesh")
            .constructor()(CtorAsRawPtr)
            .property("material", &gobot::ArrayMesh::GetMaterial, &gobot::ArrayMesh::SetMaterial)(
                    AddMetaPropertyInfo(
                            gobot::PropertyInfo()
                                .SetName("Mesh Material")
                                .SetUsageFlags(gobot::PropertyUsageFlags::Storage | gobot::PropertyUsageFlags::Editor)));

    gobot::Type::register_wrapper_converter_for_base_classes<gobot::Ref<gobot::ArrayMesh>, gobot::Ref<gobot::Mesh>>();
};
