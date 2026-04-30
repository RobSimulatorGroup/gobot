/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/resources/array_mesh.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

ArrayMesh::ArrayMesh() {
    mesh_ = RenderServer::GetInstance()->MeshCreate();
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

RID ArrayMesh::GetRid() const {
    return mesh_;
}

void ArrayMesh::UploadSurface() {
    RS::GetInstance()->MeshSetSurface(mesh_, vertices_, indices_, normals_);
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::ArrayMesh>("ArrayMesh")
            .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<gobot::Ref<gobot::ArrayMesh>, gobot::Ref<gobot::Mesh>>();
};
