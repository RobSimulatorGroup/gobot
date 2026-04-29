/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/mesh_instance_3d.hpp"

#include "gobot/core/registration.hpp"

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

}

GOBOT_REGISTRATION {

    Class_<MeshInstance3D>("MeshInstance3D")
            .constructor()(CtorAsRawPtr)
            .property("mesh", &MeshInstance3D::GetMesh, &MeshInstance3D::SetMesh)
            .property("surface_color", &MeshInstance3D::GetSurfaceColor, &MeshInstance3D::SetSurfaceColor);

};
