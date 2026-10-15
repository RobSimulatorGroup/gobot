/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/surface_mesh.hpp"
#include <algorithm>

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

void SurfaceMesh::SetUV0(const std::vector<Vector2>& uv) {
    uv0_ = uv;
    MarkChanged();
}

const std::vector<Vector2>& SurfaceMesh::GetUV0() const {
    return uv0_;
}

std::size_t SurfaceMesh::GetVertexCount() const {
    return vertices_.size();
}

std::size_t SurfaceMesh::GetTriangleCount() const {
    return triangles_.size() / 3;
}

bool SurfaceMesh::Validate(std::string* error) const {
    if ((!uv0_.empty() && uv0_.size() != vertices_.size()) ||
        !std::ranges::all_of(uv0_, [](const Vector2& uv) { return uv.allFinite(); })) {
        if (error) *error = "Surface mesh UVs must be finite and match the vertex count.";
        return false;
    }
    return ValidateTriangleMesh(vertices_, triangles_, error);
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::SurfaceMesh>("SurfaceMesh")
            .constructor()(CtorAsRawPtr)
            .property("vertices", &gobot::SurfaceMesh::GetVertices,
                      &gobot::SurfaceMesh::SetVertices)
            .property("triangles", &gobot::SurfaceMesh::GetTriangles,
                      &gobot::SurfaceMesh::SetTriangles)
            .property("uv0", &gobot::SurfaceMesh::GetUV0, &gobot::SurfaceMesh::SetUV0);

    gobot::Type::register_wrapper_converter_for_base_classes<
            gobot::Ref<gobot::SurfaceMesh>, gobot::Ref<gobot::Resource>>();
};
