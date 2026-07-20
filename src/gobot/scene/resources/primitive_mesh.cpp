/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/primitive_mesh.hpp"

#include "gobot/core/math/math_defs.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace gobot {

PrimitiveMesh::PrimitiveMesh() {
    if (RenderServer::HasInstance()) {
        mesh_ = RenderServer::GetInstance()->MeshCreate();
    }
}

PrimitiveMesh::~PrimitiveMesh() {
    if (RenderServer::HasInstance() && mesh_.IsValid()) {
        RS::GetInstance()->Free(mesh_);
    }
}

void PrimitiveMesh::SetMaterial(const Ref<Material>& material) {
    if (material_.Get() == material.Get()) {
        return;
    }
    material_ = material;
    MeshSurfaceList surfaces = GetSurfaceData() ? *GetSurfaceData() : MeshSurfaceList{};
    for (MeshSurfaceData& surface : surfaces) {
        surface.material = material_;
    }
    ReplaceSurfaceData(std::move(surfaces));
}

const Ref<Material>& PrimitiveMesh::GetMaterial() const {
    return material_;
}

RID PrimitiveMesh::GetRid() const {
    if (mesh_.IsNull() && RenderServer::HasInstance()) {
        UploadSurface();
    }
    return mesh_;
}

RID PrimitiveMesh::EnsureRid() const {
    if (mesh_.IsNull() && RenderServer::HasInstance()) {
        mesh_ = RenderServer::GetInstance()->MeshCreate();
    }
    return mesh_;
}

void PrimitiveMesh::SetGeneratedSurface(MeshSurfaceData surface) const {
    surface.material = material_;
    const_cast<PrimitiveMesh*>(this)->ReplaceSurfaceData({std::move(surface)});
    UploadSurface();
}

void PrimitiveMesh::UploadSurface() const {
    if (!RenderServer::HasInstance()) {
        return;
    }
    const auto surfaces = GetSurfaceData();
    if (!surfaces || surfaces->empty()) {
        return;
    }
    const MeshSurfaceData& surface = surfaces->front();
    RS::GetInstance()->MeshSetSurface(EnsureRid(),
                                      surface.vertices,
                                      surface.indices,
                                      surface.normals,
                                      surface.colors);
}

BoxMesh::BoxMesh() {
    UpdateMesh();
}

RID BoxMesh::GetRid() const {
    if (PrimitiveMesh::GetRid().IsNull()) {
        UpdateMesh();
    }
    return PrimitiveMesh::GetRid();
}

void BoxMesh::SetWidth(RealType width) {
    SetSize({size_.x(), std::max<RealType>(0.0, width), size_.z()});
}

RealType BoxMesh::GetWidth() const {
    return size_.y();
}

void BoxMesh::SetSize(Vector3 size) {
    size = size.cwiseMax(Vector3::Zero());
    if (size_.isApprox(size, CMP_EPSILON)) {
        return;
    }
    size_ = size;
    UpdateMesh();
}

const Vector3& BoxMesh::GetSize() const {
    return size_;
}

void BoxMesh::UpdateMesh() const {
    const Vector3 half = size_ * 0.5;
    const std::array<Vector3, 8> points = {
            Vector3{-half.x(), -half.y(), -half.z()},
            Vector3{half.x(), -half.y(), -half.z()},
            Vector3{half.x(), half.y(), -half.z()},
            Vector3{-half.x(), half.y(), -half.z()},
            Vector3{-half.x(), -half.y(), half.z()},
            Vector3{half.x(), -half.y(), half.z()},
            Vector3{half.x(), half.y(), half.z()},
            Vector3{-half.x(), half.y(), half.z()},
    };

    struct Face {
        std::array<std::uint32_t, 4> corners;
        Vector3 normal;
    };
    const std::array<Face, 6> faces = {
            Face{{0, 3, 2, 1}, Vector3{0.0, 0.0, -1.0}},
            Face{{4, 5, 6, 7}, Vector3{0.0, 0.0, 1.0}},
            Face{{0, 1, 5, 4}, Vector3{0.0, -1.0, 0.0}},
            Face{{3, 7, 6, 2}, Vector3{0.0, 1.0, 0.0}},
            Face{{1, 2, 6, 5}, Vector3{1.0, 0.0, 0.0}},
            Face{{0, 4, 7, 3}, Vector3{-1.0, 0.0, 0.0}},
    };
    const std::array<Vector2, 4> uv = {
            Vector2{0.0, 0.0}, Vector2{1.0, 0.0}, Vector2{1.0, 1.0}, Vector2{0.0, 1.0}};

    MeshSurfaceData surface;
    surface.vertices.reserve(24);
    surface.normals.reserve(24);
    surface.uv0.reserve(24);
    surface.indices.reserve(36);
    for (const Face& face : faces) {
        const std::uint32_t base = static_cast<std::uint32_t>(surface.vertices.size());
        for (std::size_t i = 0; i < face.corners.size(); ++i) {
            surface.vertices.push_back(points[face.corners[i]]);
            surface.normals.push_back(face.normal);
            surface.uv0.push_back(uv[i]);
        }
        surface.indices.insert(surface.indices.end(),
                               {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    SetGeneratedSurface(std::move(surface));
}

CylinderMesh::CylinderMesh() {
    UpdateMesh();
}

RID CylinderMesh::GetRid() const {
    if (PrimitiveMesh::GetRid().IsNull()) {
        UpdateMesh();
    }
    return PrimitiveMesh::GetRid();
}

void CylinderMesh::SetRadius(RealType radius) {
    if (radius < 0.0) {
        LOG_ERROR("CylinderMesh radius cannot be negative.");
        return;
    }
    if (radius_ != radius) {
        radius_ = radius;
        UpdateMesh();
    }
}

RealType CylinderMesh::GetRadius() const { return radius_; }

void CylinderMesh::SetHeight(RealType height) {
    if (height < 0.0) {
        LOG_ERROR("CylinderMesh height cannot be negative.");
        return;
    }
    if (height_ != height) {
        height_ = height;
        UpdateMesh();
    }
}

RealType CylinderMesh::GetHeight() const { return height_; }

void CylinderMesh::SetRadialSegments(int radial_segments) {
    radial_segments = std::max(radial_segments, 3);
    if (radial_segments_ != radial_segments) {
        radial_segments_ = radial_segments;
        UpdateMesh();
    }
}

int CylinderMesh::GetRadialSegments() const { return radial_segments_; }

void CylinderMesh::UpdateMesh() const {
    MeshSurfaceData surface;
    const RealType half_height = height_ * 0.5;

    for (int segment = 0; segment <= radial_segments_; ++segment) {
        const RealType u = static_cast<RealType>(segment) / radial_segments_;
        const RealType angle = 2.0 * Math_PI * u;
        const RealType x = std::cos(angle) * radius_;
        const RealType z = std::sin(angle) * radius_;
        const Vector3 normal{std::cos(angle), 0.0, std::sin(angle)};
        surface.vertices.emplace_back(x, half_height, z);
        surface.vertices.emplace_back(x, -half_height, z);
        surface.normals.push_back(normal);
        surface.normals.push_back(normal);
        surface.uv0.emplace_back(u, 1.0);
        surface.uv0.emplace_back(u, 0.0);
    }
    for (int segment = 0; segment < radial_segments_; ++segment) {
        const std::uint32_t top = static_cast<std::uint32_t>(segment * 2);
        const std::uint32_t bottom = top + 1;
        const std::uint32_t next_top = top + 2;
        const std::uint32_t next_bottom = top + 3;
        surface.indices.insert(surface.indices.end(),
                               {top, next_top, next_bottom, top, next_bottom, bottom});
    }

    const std::uint32_t top_center = static_cast<std::uint32_t>(surface.vertices.size());
    surface.vertices.emplace_back(0.0, half_height, 0.0);
    surface.normals.push_back(Vector3::UnitY());
    surface.uv0.emplace_back(0.5, 0.5);
    const std::uint32_t top_ring = static_cast<std::uint32_t>(surface.vertices.size());
    for (int segment = 0; segment <= radial_segments_; ++segment) {
        const RealType angle = 2.0 * Math_PI * segment / radial_segments_;
        const RealType x = std::cos(angle) * radius_;
        const RealType z = std::sin(angle) * radius_;
        surface.vertices.emplace_back(x, half_height, z);
        surface.normals.push_back(Vector3::UnitY());
        surface.uv0.emplace_back(0.5 + 0.5 * std::cos(angle), 0.5 + 0.5 * std::sin(angle));
    }
    for (int segment = 0; segment < radial_segments_; ++segment) {
        const std::uint32_t current = top_ring + static_cast<std::uint32_t>(segment);
        surface.indices.insert(surface.indices.end(), {top_center, current + 1, current});
    }

    const std::uint32_t bottom_center = static_cast<std::uint32_t>(surface.vertices.size());
    surface.vertices.emplace_back(0.0, -half_height, 0.0);
    surface.normals.push_back(-Vector3::UnitY());
    surface.uv0.emplace_back(0.5, 0.5);
    const std::uint32_t bottom_ring = static_cast<std::uint32_t>(surface.vertices.size());
    for (int segment = 0; segment <= radial_segments_; ++segment) {
        const RealType angle = 2.0 * Math_PI * segment / radial_segments_;
        const RealType x = std::cos(angle) * radius_;
        const RealType z = std::sin(angle) * radius_;
        surface.vertices.emplace_back(x, -half_height, z);
        surface.normals.push_back(-Vector3::UnitY());
        surface.uv0.emplace_back(0.5 + 0.5 * std::cos(angle), 0.5 + 0.5 * std::sin(angle));
    }
    for (int segment = 0; segment < radial_segments_; ++segment) {
        const std::uint32_t current = bottom_ring + static_cast<std::uint32_t>(segment);
        surface.indices.insert(surface.indices.end(), {bottom_center, current, current + 1});
    }

    SetGeneratedSurface(std::move(surface));
}

PlaneMesh::PlaneMesh() {
    UpdateMesh();
}

RID PlaneMesh::GetRid() const {
    if (PrimitiveMesh::GetRid().IsNull()) {
        UpdateMesh();
    }
    return PrimitiveMesh::GetRid();
}

void PlaneMesh::SetSize(Vector2 size) {
    size = size.cwiseMax(Vector2::Zero());
    if (!size_.isApprox(size, CMP_EPSILON)) {
        size_ = size;
        UpdateMesh();
    }
}

const Vector2& PlaneMesh::GetSize() const { return size_; }

void PlaneMesh::UpdateMesh() const {
    const RealType half_x = size_.x() * 0.5;
    const RealType half_y = size_.y() * 0.5;
    MeshSurfaceData surface;
    surface.vertices = {
            {-half_x, -half_y, 0.0}, {half_x, -half_y, 0.0},
            {half_x, half_y, 0.0}, {-half_x, half_y, 0.0}};
    surface.indices = {0, 1, 2, 0, 2, 3};
    surface.normals.assign(4, Vector3::UnitZ());
    surface.uv0 = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
    SetGeneratedSurface(std::move(surface));
}

SphereMesh::SphereMesh() {
    UpdateMesh();
}

RID SphereMesh::GetRid() const {
    if (PrimitiveMesh::GetRid().IsNull()) {
        UpdateMesh();
    }
    return PrimitiveMesh::GetRid();
}

void SphereMesh::SetRadius(RealType radius) {
    if (radius < 0.0) {
        LOG_ERROR("SphereMesh radius cannot be negative.");
        return;
    }
    if (radius_ != radius) {
        radius_ = radius;
        UpdateMesh();
    }
}

RealType SphereMesh::GetRadius() const { return radius_; }

void SphereMesh::SetRadialSegments(int radial_segments) {
    radial_segments = std::max(radial_segments, 3);
    if (radial_segments_ != radial_segments) {
        radial_segments_ = radial_segments;
        UpdateMesh();
    }
}

int SphereMesh::GetRadialSegments() const { return radial_segments_; }

void SphereMesh::SetRings(int rings) {
    rings = std::max(rings, 2);
    if (rings_ != rings) {
        rings_ = rings;
        UpdateMesh();
    }
}

int SphereMesh::GetRings() const { return rings_; }

void SphereMesh::UpdateMesh() const {
    MeshSurfaceData surface;
    surface.vertices.reserve(static_cast<std::size_t>(rings_ + 1) * (radial_segments_ + 1));
    surface.normals.reserve(surface.vertices.capacity());
    surface.uv0.reserve(surface.vertices.capacity());

    for (int ring = 0; ring <= rings_; ++ring) {
        const RealType v = static_cast<RealType>(ring) / rings_;
        const RealType theta = Math_PI * v;
        const RealType y = std::cos(theta);
        const RealType ring_radius = std::sin(theta);
        for (int segment = 0; segment <= radial_segments_; ++segment) {
            const RealType u = static_cast<RealType>(segment) / radial_segments_;
            const RealType phi = 2.0 * Math_PI * u;
            const Vector3 normal{std::cos(phi) * ring_radius, y, std::sin(phi) * ring_radius};
            surface.vertices.push_back(normal * radius_);
            surface.normals.push_back(normal);
            surface.uv0.emplace_back(u, v);
        }
    }

    const int stride = radial_segments_ + 1;
    for (int ring = 0; ring < rings_; ++ring) {
        for (int segment = 0; segment < radial_segments_; ++segment) {
            const std::uint32_t a = static_cast<std::uint32_t>(ring * stride + segment);
            const std::uint32_t b = static_cast<std::uint32_t>((ring + 1) * stride + segment);
            const std::uint32_t c = a + 1;
            const std::uint32_t d = b + 1;
            surface.indices.insert(surface.indices.end(), {a, b, c, c, b, d});
        }
    }
    SetGeneratedSurface(std::move(surface));
}

CapsuleMesh::CapsuleMesh() = default;

} // namespace gobot

GOBOT_REGISTRATION {
    USING_ENUM_BITWISE_OPERATORS;

    Class_<PrimitiveMesh>("PrimitiveMesh")
            .constructor()(CtorAsRawPtr)
            .property("material", &PrimitiveMesh::GetMaterial, &PrimitiveMesh::SetMaterial)(
                    AddMetaPropertyInfo(
                            PropertyInfo()
                                    .SetName("Mesh Material")
                                    .SetUsageFlags(PropertyUsageFlags::Storage | PropertyUsageFlags::Editor)));
    Type::register_wrapper_converter_for_base_classes<Ref<PrimitiveMesh>, Ref<Mesh>>();

    Class_<BoxMesh>("BoxMesh")
            .constructor()(CtorAsRawPtr)
            .property("size", &BoxMesh::GetSize, &BoxMesh::SetSize);
    Type::register_wrapper_converter_for_base_classes<Ref<BoxMesh>, Ref<PrimitiveMesh>>();

    Class_<CylinderMesh>("CylinderMesh")
            .constructor()(CtorAsRawPtr)
            .property("radius", &CylinderMesh::GetRadius, &CylinderMesh::SetRadius)
            .property("height", &CylinderMesh::GetHeight, &CylinderMesh::SetHeight)
            .property("radial_segments", &CylinderMesh::GetRadialSegments, &CylinderMesh::SetRadialSegments);
    Type::register_wrapper_converter_for_base_classes<Ref<CylinderMesh>, Ref<PrimitiveMesh>>();

    Class_<PlaneMesh>("PlaneMesh")
            .constructor()(CtorAsRawPtr)
            .property("size", &PlaneMesh::GetSize, &PlaneMesh::SetSize);
    Type::register_wrapper_converter_for_base_classes<Ref<PlaneMesh>, Ref<PrimitiveMesh>>();

    Class_<SphereMesh>("SphereMesh")
            .constructor()(CtorAsRawPtr)
            .property("radius", &SphereMesh::GetRadius, &SphereMesh::SetRadius)
            .property("radial_segments", &SphereMesh::GetRadialSegments, &SphereMesh::SetRadialSegments)
            .property("rings", &SphereMesh::GetRings, &SphereMesh::SetRings);
    Type::register_wrapper_converter_for_base_classes<Ref<SphereMesh>, Ref<PrimitiveMesh>>();
}
