/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/mesh_instance_3d.hpp"

#include "gobot/core/registration.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"

namespace gobot {

void MeshInstance3D::SetMesh(const Ref<Mesh>& mesh) {
    mesh_ = mesh;
}

const Ref<Mesh>& MeshInstance3D::GetMesh() const {
    return mesh_;
}

void MeshInstance3D::SetSurfaceColor(const Color& color) {
    surface_color_ = color;
}

Color MeshInstance3D::GetSurfaceColor() const {
    return surface_color_;
}

void MeshInstance3D::SetMaterial(const Ref<Material>& material) {
    material_ = material;
}

const Ref<Material>& MeshInstance3D::GetMaterial() const {
    return material_;
}

void MeshInstance3D::SetMeshMaterial(const Ref<Material>& material) {
    if (Ref<ArrayMesh> array_mesh = dynamic_pointer_cast<ArrayMesh>(mesh_); array_mesh.IsValid()) {
        array_mesh->SetMaterial(material);
        return;
    }

    if (Ref<PrimitiveMesh> primitive_mesh = dynamic_pointer_cast<PrimitiveMesh>(mesh_); primitive_mesh.IsValid()) {
        primitive_mesh->SetMaterial(material);
    }
}

Ref<Material> MeshInstance3D::GetMeshMaterial() const {
    if (Ref<ArrayMesh> array_mesh = dynamic_pointer_cast<ArrayMesh>(mesh_); array_mesh.IsValid()) {
        return array_mesh->GetMaterial();
    }

    if (Ref<PrimitiveMesh> primitive_mesh = dynamic_pointer_cast<PrimitiveMesh>(mesh_); primitive_mesh.IsValid()) {
        return primitive_mesh->GetMaterial();
    }

    return {};
}

Ref<Material> MeshInstance3D::GetActiveMaterial() const {
    if (material_.IsValid()) {
        return material_;
    }

    return GetMeshMaterial();
}

}

GOBOT_REGISTRATION {

    USING_ENUM_BITWISE_OPERATORS;

    Class_<MeshInstance3D>("MeshInstance3D")
            .constructor()(CtorAsRawPtr)
            .property("mesh", &MeshInstance3D::GetMesh, &MeshInstance3D::SetMesh)
            .property("material", &MeshInstance3D::GetMaterial, &MeshInstance3D::SetMaterial)(
                    AddMetaPropertyInfo(
                            PropertyInfo()
                                .SetName("Material")
                                .SetUsageFlags(PropertyUsageFlags::Storage | PropertyUsageFlags::Editor)))
            .property("mesh_material", &MeshInstance3D::GetMeshMaterial, &MeshInstance3D::SetMeshMaterial)(
                    AddMetaPropertyInfo(
                            PropertyInfo()
                                .SetUsageFlags(PropertyUsageFlags::Storage)))
            .property("surface_color", &MeshInstance3D::GetSurfaceColor, &MeshInstance3D::SetSurfaceColor);

};
