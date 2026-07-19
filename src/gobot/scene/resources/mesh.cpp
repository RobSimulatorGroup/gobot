/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-12-21
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/mesh.hpp"
#include "gobot/core/math/math_defs.hpp"
#include "gobot/core/registration.hpp"

#include <algorithm>
#include <cmath>

namespace gobot {

namespace {

Vector3 SafeNormal(const Vector3& value, const Vector3& fallback) {
    const RealType length = value.norm();
    if (length <= CMP_EPSILON || !value.allFinite()) {
        return fallback;
    }
    return value / length;
}

Vector3 DefaultTangentForNormal(const Vector3& normal) {
    const Vector3 helper = std::abs(normal.z()) < 0.999 ? Vector3::UnitZ() : Vector3::UnitY();
    return SafeNormal(helper.cross(normal), Vector3::UnitX());
}

} // namespace

void CompleteMeshSurface(MeshSurfaceData& surface) {
    const std::size_t vertex_count = surface.vertices.size();
    if (vertex_count == 0) {
        surface.indices.clear();
        surface.normals.clear();
        surface.tangents.clear();
        surface.uv0.clear();
        surface.colors.clear();
        return;
    }

    surface.indices.erase(
            std::remove_if(surface.indices.begin(), surface.indices.end(),
                           [vertex_count](std::uint32_t index) { return index >= vertex_count; }),
            surface.indices.end());
    surface.indices.resize(surface.indices.size() - surface.indices.size() % 3);

    if (surface.normals.size() != vertex_count) {
        surface.normals.assign(vertex_count, Vector3::Zero());
        for (std::size_t i = 0; i + 2 < surface.indices.size(); i += 3) {
            const std::uint32_t ia = surface.indices[i];
            const std::uint32_t ib = surface.indices[i + 1];
            const std::uint32_t ic = surface.indices[i + 2];
            const Vector3 face = (surface.vertices[ib] - surface.vertices[ia])
                                         .cross(surface.vertices[ic] - surface.vertices[ia]);
            if (face.squaredNorm() <= CMP_EPSILON2) {
                continue;
            }
            surface.normals[ia] += face;
            surface.normals[ib] += face;
            surface.normals[ic] += face;
        }
        for (Vector3& normal : surface.normals) {
            normal = SafeNormal(normal, Vector3::UnitZ());
        }
    } else {
        for (Vector3& normal : surface.normals) {
            normal = SafeNormal(normal, Vector3::UnitZ());
        }
    }

    if (!surface.uv0.empty() && surface.uv0.size() != vertex_count) {
        surface.uv0.clear();
    }
    if (!surface.colors.empty() && surface.colors.size() != vertex_count) {
        surface.colors.clear();
    }

    if (surface.tangents.size() != vertex_count) {
        if (surface.uv0.size() != vertex_count) {
            surface.tangents.clear();
            return;
        }
        std::vector<Vector3> tangent_sum(vertex_count, Vector3::Zero());
        std::vector<Vector3> bitangent_sum(vertex_count, Vector3::Zero());
        for (std::size_t i = 0; i + 2 < surface.indices.size(); i += 3) {
            const std::uint32_t ia = surface.indices[i];
            const std::uint32_t ib = surface.indices[i + 1];
            const std::uint32_t ic = surface.indices[i + 2];
            const Vector3 edge1 = surface.vertices[ib] - surface.vertices[ia];
            const Vector3 edge2 = surface.vertices[ic] - surface.vertices[ia];
            const Vector2 duv1 = surface.uv0[ib] - surface.uv0[ia];
            const Vector2 duv2 = surface.uv0[ic] - surface.uv0[ia];
            const RealType determinant = duv1.x() * duv2.y() - duv1.y() * duv2.x();
            if (std::abs(determinant) <= CMP_EPSILON) {
                continue;
            }
            const RealType reciprocal = 1.0 / determinant;
            const Vector3 tangent = (edge1 * duv2.y() - edge2 * duv1.y()) * reciprocal;
            const Vector3 bitangent = (edge2 * duv1.x() - edge1 * duv2.x()) * reciprocal;
            tangent_sum[ia] += tangent;
            tangent_sum[ib] += tangent;
            tangent_sum[ic] += tangent;
            bitangent_sum[ia] += bitangent;
            bitangent_sum[ib] += bitangent;
            bitangent_sum[ic] += bitangent;
        }

        surface.tangents.resize(vertex_count);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            const Vector3& normal = surface.normals[i];
            Vector3 tangent = tangent_sum[i] - normal * normal.dot(tangent_sum[i]);
            tangent = SafeNormal(tangent, DefaultTangentForNormal(normal));
            const RealType handedness = normal.cross(tangent).dot(bitangent_sum[i]) < 0.0 ? -1.0 : 1.0;
            surface.tangents[i] = Vector4{tangent.x(), tangent.y(), tangent.z(), handedness};
        }
    }
}

Mesh::Mesh()
    : surface_data_(std::make_shared<const MeshSurfaceList>()) {

}

std::shared_ptr<const MeshSurfaceList> Mesh::GetSurfaceData() const {
    return surface_data_;
}

std::size_t Mesh::GetSurfaceCount() const {
    return surface_data_ ? surface_data_->size() : 0;
}

void Mesh::ReplaceSurfaceData(MeshSurfaceList surfaces) {
    for (MeshSurfaceData& surface : surfaces) {
        CompleteMeshSurface(surface);
    }
    surface_data_ = std::make_shared<const MeshSurfaceList>(std::move(surfaces));
    MarkChanged();

}

}

GOBOT_REGISTRATION {
    Class_<MeshSurfaceData>("MeshSurfaceData")
            .constructor()(CtorAsObject)
            .property("vertices", &MeshSurfaceData::vertices)
            .property("indices", &MeshSurfaceData::indices)
            .property("normals", &MeshSurfaceData::normals)
            .property("tangents", &MeshSurfaceData::tangents)
            .property("uv0", &MeshSurfaceData::uv0)
            .property("colors", &MeshSurfaceData::colors)
            .property("material", &MeshSurfaceData::material);

    Class_<Mesh>("Mesh")
        .constructor()(CtorAsRawPtr);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<Mesh>, Ref<Resource>>();

};
