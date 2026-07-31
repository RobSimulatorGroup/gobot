/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

class GOBOT_EXPORT ConvexMeshShape3D : public Shape3D {
    GOBCLASS(ConvexMeshShape3D, Shape3D);
public:
    ConvexMeshShape3D();

    void SetMesh(const Ref<Mesh>& mesh);

    const Ref<Mesh>& GetMesh() const;

private:
    Ref<Mesh> mesh_{nullptr};
};

} // namespace gobot
