/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
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
                           std::vector<Vector3> normals,
                           std::vector<Color> colors,
                           std::vector<Vector4> tangents,
                           std::vector<Vector2> uv0) {
    MeshSurfaceData surface;
    surface.vertices = std::move(vertices);
    surface.indices = std::move(indices);
    surface.normals = std::move(normals);
    surface.colors = std::move(colors);
    surface.tangents = std::move(tangents);
    surface.uv0 = std::move(uv0);
    surface.material = material_;
    ReplaceSurfaceData({std::move(surface)});
    UploadSurface();
}

void ArrayMesh::SetSurfaces(MeshSurfaceList surfaces) {
    ReplaceSurfaceData(std::move(surfaces));
    UploadSurface();
}

MeshSurfaceList ArrayMesh::GetSurfaces() const {
    const auto surfaces = GetSurfaceData();
    return surfaces ? *surfaces : MeshSurfaceList{};
}

const std::vector<Vector3>& ArrayMesh::GetVertices() const {
    static const std::vector<Vector3> empty;
    const auto surfaces = GetSurfaceData();
    return surfaces && !surfaces->empty() ? surfaces->front().vertices : empty;
}

const std::vector<uint32_t>& ArrayMesh::GetIndices() const {
    static const std::vector<uint32_t> empty;
    const auto surfaces = GetSurfaceData();
    return surfaces && !surfaces->empty() ? surfaces->front().indices : empty;
}

const std::vector<Vector3>& ArrayMesh::GetNormals() const {
    static const std::vector<Vector3> empty;
    const auto surfaces = GetSurfaceData();
    return surfaces && !surfaces->empty() ? surfaces->front().normals : empty;
}

const std::vector<Color>& ArrayMesh::GetColors() const {
    static const std::vector<Color> empty;
    const auto surfaces = GetSurfaceData();
    return surfaces && !surfaces->empty() ? surfaces->front().colors : empty;
}

void ArrayMesh::SetMaterial(const Ref<Material>& material) {
    if (material_.Get() == material.Get()) {
        return;
    }
    material_ = material;
    MarkChanged();
}

const Ref<Material>& ArrayMesh::GetMaterial() const {
    return material_;
}

RID ArrayMesh::GetRid() const {
    if (mesh_.IsNull() && RenderServer::HasInstance() && !GetVertices().empty() && !GetIndices().empty()) {
        UploadSurface();
    }
    return mesh_;
}

void ArrayMesh::UploadSurface() const {
    if (!RenderServer::HasInstance()) {
        return;
    }
    if (mesh_.IsNull()) {
        mesh_ = RenderServer::GetInstance()->MeshCreate();
    }
    RS::GetInstance()->MeshSetSurface(mesh_, GetVertices(), GetIndices(), GetNormals(), GetColors());
}

} // namespace gobot

GOBOT_REGISTRATION {
    USING_ENUM_BITWISE_OPERATORS;

    Class_<gobot::ArrayMesh>("ArrayMesh")
            .constructor()(CtorAsRawPtr)
            .property("surfaces", &gobot::ArrayMesh::GetSurfaces, &gobot::ArrayMesh::SetSurfaces)
            .property("material", &gobot::ArrayMesh::GetMaterial, &gobot::ArrayMesh::SetMaterial)(
                    AddMetaPropertyInfo(
                            gobot::PropertyInfo()
                                .SetName("Mesh Material")
                                .SetUsageFlags(gobot::PropertyUsageFlags::Storage | gobot::PropertyUsageFlags::Editor)));

    gobot::Type::register_wrapper_converter_for_base_classes<gobot::Ref<gobot::ArrayMesh>, gobot::Ref<gobot::Mesh>>();
};
