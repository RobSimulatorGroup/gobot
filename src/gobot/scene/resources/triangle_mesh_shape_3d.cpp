/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/triangle_mesh_shape_3d.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

TriangleMeshShape3D::TriangleMeshShape3D() = default;

void TriangleMeshShape3D::SetMesh(const Ref<Mesh> &mesh) {
    if (mesh_.Get() == mesh.Get()) {
        return;
    }
    mesh_ = mesh;
    MarkChanged();
}

const Ref<Mesh> &TriangleMeshShape3D::GetMesh() const { return mesh_; }

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::TriangleMeshShape3D>("TriangleMeshShape3D")
        .constructor()(CtorAsRawPtr)
        .property("mesh", &gobot::TriangleMeshShape3D::GetMesh, &gobot::TriangleMeshShape3D::SetMesh);

    gobot::Type::register_wrapper_converter_for_base_classes<gobot::Ref<gobot::TriangleMeshShape3D>,
                                                             gobot::Ref<gobot::Shape3D>>();
};
