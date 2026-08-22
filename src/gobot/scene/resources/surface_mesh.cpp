/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/surface_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>

#include "gobot/core/registration.hpp"

namespace gobot {
namespace {

bool SetValidationError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

} // namespace

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
    if (vertices_.size() < 3) {
        return SetValidationError(error, "surface mesh requires at least three vertices");
    }
    for (std::size_t index = 0; index < vertices_.size(); ++index) {
        if (!vertices_[index].allFinite()) {
            return SetValidationError(
                    error, "surface mesh vertex " + std::to_string(index) + " is not finite");
        }
    }
    if (triangles_.empty() || triangles_.size() % 3 != 0) {
        return SetValidationError(
                error, "surface mesh indices must contain one or more groups of three");
    }

    std::set<std::array<std::uint32_t, 3>> unique_triangles;
    std::vector<bool> referenced(vertices_.size(), false);
    for (std::size_t index = 0; index < triangles_.size(); index += 3) {
        const std::array<std::uint32_t, 3> triangle{
                triangles_[index], triangles_[index + 1], triangles_[index + 2]};
        for (const std::uint32_t vertex : triangle) {
            if (vertex >= vertices_.size()) {
                return SetValidationError(
                        error, "surface mesh references vertex outside its vertex table");
            }
            referenced[vertex] = true;
        }
        std::array<std::uint32_t, 3> sorted = triangle;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            return SetValidationError(error, "surface mesh contains a repeated triangle vertex");
        }
        if (!unique_triangles.insert(sorted).second) {
            return SetValidationError(error, "surface mesh contains a duplicate triangle");
        }
        const Vector3& a = vertices_[triangle[0]];
        const Vector3& b = vertices_[triangle[1]];
        const Vector3& c = vertices_[triangle[2]];
        const RealType edge_scale = std::max({
                (b - a).norm(), (c - a).norm(), (c - b).norm()});
        const RealType area_tolerance =
                std::numeric_limits<RealType>::epsilon() * 128.0 *
                edge_scale * edge_scale;
        const RealType double_area = (b - a).cross(c - a).norm();
        if (!std::isfinite(double_area) || !std::isfinite(area_tolerance) ||
            edge_scale <= 0.0 || double_area <= area_tolerance) {
            return SetValidationError(
                    error, "surface mesh requires finite, non-degenerate triangles");
        }
    }
    if (std::find(referenced.begin(), referenced.end(), false) != referenced.end()) {
        return SetValidationError(
                error, "surface mesh contains a vertex not referenced by any triangle");
    }
    if (error != nullptr) {
        error->clear();
    }
    return true;
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
