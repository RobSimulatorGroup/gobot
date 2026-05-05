/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/


#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

#include <algorithm>

namespace gobot {

PrimitiveMesh::PrimitiveMesh()
{
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
    material_ = material;
}

const Ref<Material>& PrimitiveMesh::GetMaterial() const {
    return material_;
}

RID PrimitiveMesh::GetRid() const {
    return mesh_;
}


//////////////////////

BoxMesh::BoxMesh() {
    if (RenderServer::HasInstance()) {
        RS::GetInstance()->MeshSetBox(GetRid(), size_);
    }
}

void BoxMesh::SetWidth(RealType p_width) {
    size_[1] = p_width;
}

RealType BoxMesh::GetWidth() const {
    return size_[1];
}


void BoxMesh::SetSize(Vector3 size) {
    size_ = size;
    if (RenderServer::HasInstance()) {
        RS::GetInstance()->MeshSetBox(GetRid(), size_);
    }
}

const Vector3& BoxMesh::GetSize() const {
    return size_;
}

/////////////////////////////////

CylinderMesh::CylinderMesh() {
    UpdateMesh();
}

void CylinderMesh::SetRadius(RealType radius) {
    if (radius < 0) {
        LOG_ERROR("CylinderMesh radius cannot be negative.");
        return;
    }
    radius_ = radius;
    UpdateMesh();
}

RealType CylinderMesh::GetRadius() const {
    return radius_;
}

void CylinderMesh::SetHeight(RealType height) {
    if (height < 0) {
        LOG_ERROR("CylinderMesh height cannot be negative.");
        return;
    }
    height_ = height;
    UpdateMesh();
}

RealType CylinderMesh::GetHeight() const {
    return height_;
}

void CylinderMesh::SetRadialSegments(int radial_segments) {
    radial_segments_ = std::max(radial_segments, 3);
    UpdateMesh();
}

int CylinderMesh::GetRadialSegments() const {
    return radial_segments_;
}

void CylinderMesh::UpdateMesh() {
    if (!RenderServer::HasInstance()) {
        return;
    }
    RS::GetInstance()->MeshSetCylinder(GetRid(), radius_, height_, radial_segments_);
}

/////////////////////////////////

PlaneMesh::PlaneMesh() {

}

/////////////////////////////////

SphereMesh::SphereMesh() {
    UpdateMesh();
}

void SphereMesh::SetRadius(RealType radius) {
    if (radius < 0) {
        LOG_ERROR("SphereMesh radius cannot be negative.");
        return;
    }
    radius_ = radius;
    UpdateMesh();
}

RealType SphereMesh::GetRadius() const {
    return radius_;
}

void SphereMesh::SetRadialSegments(int radial_segments) {
    radial_segments_ = std::max(radial_segments, 3);
    UpdateMesh();
}

int SphereMesh::GetRadialSegments() const {
    return radial_segments_;
}

void SphereMesh::SetRings(int rings) {
    rings_ = std::max(rings, 2);
    UpdateMesh();
}

int SphereMesh::GetRings() const {
    return rings_;
}

void SphereMesh::UpdateMesh() {
    if (!RenderServer::HasInstance()) {
        return;
    }
    RS::GetInstance()->MeshSetSphere(GetRid(), radius_, radial_segments_, rings_);
}

/////////////////////////////////

CapsuleMesh::CapsuleMesh() {

}

} // end of namespace gobot


GOBOT_REGISTRATION {
    USING_ENUM_BITWISE_OPERATORS;

    Class_<PrimitiveMesh>("PrimitiveMesh")
            .constructor()(CtorAsRawPtr)
            .property("material", &PrimitiveMesh::GetMaterial, &PrimitiveMesh::SetMaterial)(
                    AddMetaPropertyInfo(
                            PropertyInfo()
                                .SetName("Mesh Material")
                                .SetUsageFlags(PropertyUsageFlags::Storage | PropertyUsageFlags::Editor)));

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PrimitiveMesh>, Ref<Mesh>>();

    Class_<BoxMesh>("BoxMesh")
            .constructor()(CtorAsRawPtr)
            .property("size", &BoxMesh::GetSize, &BoxMesh::SetSize);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<BoxMesh>, Ref<PrimitiveMesh>>();

    Class_<CylinderMesh>("CylinderMesh")
            .constructor()(CtorAsRawPtr)
            .property("radius", &CylinderMesh::GetRadius, &CylinderMesh::SetRadius)
            .property("height", &CylinderMesh::GetHeight, &CylinderMesh::SetHeight)
            .property("radial_segments", &CylinderMesh::GetRadialSegments, &CylinderMesh::SetRadialSegments);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<CylinderMesh>, Ref<PrimitiveMesh>>();

    Class_<SphereMesh>("SphereMesh")
            .constructor()(CtorAsRawPtr)
            .property("radius", &SphereMesh::GetRadius, &SphereMesh::SetRadius)
            .property("radial_segments", &SphereMesh::GetRadialSegments, &SphereMesh::SetRadialSegments)
            .property("rings", &SphereMesh::GetRings, &SphereMesh::SetRings);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<SphereMesh>, Ref<PrimitiveMesh>>();

};
