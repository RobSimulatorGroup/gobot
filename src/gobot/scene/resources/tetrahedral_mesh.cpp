/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/tetrahedral_mesh.hpp"

#include "gobot/core/math/mesh_validation.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void TetrahedralMesh::SetVertices(const std::vector<Vector3>& vertices) {
    vertices_ = vertices;
    MarkChanged();
}

const std::vector<Vector3>& TetrahedralMesh::GetVertices() const {
    return vertices_;
}

void TetrahedralMesh::SetTetrahedra(const std::vector<std::uint32_t>& tetrahedra) {
    tetrahedra_ = tetrahedra;
    MarkChanged();
}

const std::vector<std::uint32_t>& TetrahedralMesh::GetTetrahedra() const {
    return tetrahedra_;
}

void TetrahedralMesh::SetSurfaceTriangles(
        const std::vector<std::uint32_t>& surface_triangles) {
    surface_triangles_ = surface_triangles;
    MarkChanged();
}

const std::vector<std::uint32_t>& TetrahedralMesh::GetSurfaceTriangles() const {
    return surface_triangles_;
}

std::vector<std::uint32_t> TetrahedralMesh::GetResolvedSurfaceTriangles() const {
    return ResolveTetrahedralSurface(tetrahedra_, surface_triangles_);
}

std::size_t TetrahedralMesh::GetVertexCount() const {
    return vertices_.size();
}

std::size_t TetrahedralMesh::GetTetrahedronCount() const {
    return tetrahedra_.size() / 4;
}

bool TetrahedralMesh::Validate(std::string* error) const {
    return ValidateTetrahedralMesh(vertices_, tetrahedra_, surface_triangles_, error);
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::TetrahedralMesh>("TetrahedralMesh")
            .constructor()(CtorAsRawPtr)
            .property("vertices", &gobot::TetrahedralMesh::GetVertices,
                      &gobot::TetrahedralMesh::SetVertices)
            .property("tetrahedra", &gobot::TetrahedralMesh::GetTetrahedra,
                      &gobot::TetrahedralMesh::SetTetrahedra)
            .property("surface_triangles", &gobot::TetrahedralMesh::GetSurfaceTriangles,
                      &gobot::TetrahedralMesh::SetSurfaceTriangles);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::TetrahedralMesh>, gobot::Ref<gobot::Resource>>();
};
