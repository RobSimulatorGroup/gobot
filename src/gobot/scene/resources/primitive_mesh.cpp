/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/


#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/core/registration.hpp"

namespace gobot {

PrimitiveMesh::PrimitiveMesh()
{
    mesh_ = RenderServer::GetInstance()->MeshCreate();
}

PrimitiveMesh::~PrimitiveMesh() {
    RS::GetInstance()->Free(mesh_);
}

void PrimitiveMesh::SetMaterial(const Ref<Material>& material) {
    material_ = material;
}

const Ref<Material>& PrimitiveMesh::GetMaterial() const {
    return material_;
}


//////////////////////

BoxMesh::BoxMesh() {

}

void BoxMesh::SetWidth(RealType p_width) {
    size_[1] = p_width;
}

RealType BoxMesh::GetWidth() const {
    return size_[1];
}


void BoxMesh::SetSize(Vector3 size) {
    size_ = size;
}

const Vector3& BoxMesh::GetSize() const {
    return size_;
}

/////////////////////////////////

CylinderMesh::CylinderMesh() {

}

/////////////////////////////////

PlaneMesh::PlaneMesh() {

}

/////////////////////////////////

CapsuleMesh::CapsuleMesh() {

}

} // end of namespace gobot


GOBOT_REGISTRATION {
    Class_<PrimitiveMesh>("PrimitiveMesh")
            .constructor()(CtorAsRawPtr)
            .property("material", &PrimitiveMesh::GetMaterial, &PrimitiveMesh::SetMaterial);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<PrimitiveMesh>, Ref<Mesh>>();

    Class_<BoxMesh>("BoxMesh")
            .constructor()(CtorAsRawPtr)
            .property("size", &BoxMesh::GetSize, &BoxMesh::SetSize);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<BoxMesh>, Ref<PrimitiveMesh>>();

};

