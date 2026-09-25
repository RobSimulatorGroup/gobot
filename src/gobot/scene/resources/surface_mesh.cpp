/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/surface_mesh.hpp"

#include "gobot/core/math/mesh_validation.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void SurfaceMesh::SetVertices(const std::vector<Vector3>& vertices) {
    vertices_ = vertices;
    MarkChanged();
}

const std::vector<Vector3>& SurfaceMesh::GetVertices() const {
    return vertices_;
}

void SurfaceMesh::SetTriangles(const std::vector<std::uint32_t>& triangles) {
    triangles_ = triangles;
    MarkChanged();
}

const std::vector<std::uint32_t>& SurfaceMesh::GetTriangles() const {
    return triangles_;
}

std::size_t SurfaceMesh::GetVertexCount() const {
    return vertices_.size();
}

std::size_t SurfaceMesh::GetTriangleCount() const {
    return triangles_.size() / 3;
}

bool SurfaceMesh::Validate(std::string* error) const {
    return ValidateTriangleMesh(vertices_, triangles_, error);
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::SurfaceMesh>("SurfaceMesh")
            .constructor()(CtorAsRawPtr)
            .property("vertices", &gobot::SurfaceMesh::GetVertices,
                      &gobot::SurfaceMesh::SetVertices)
            .property("triangles", &gobot::SurfaceMesh::GetTriangles,
                      &gobot::SurfaceMesh::SetTriangles);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::SurfaceMesh>, gobot::Ref<gobot::Resource>>();
};
