/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/core/color.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"

namespace gobot {

class GOBOT_EXPORT MeshInstance3D : public Node3D {
    GOBCLASS(MeshInstance3D, Node3D)

public:
    MeshInstance3D() = default;

    void SetMesh(const Ref<Mesh>& mesh);

    const Ref<Mesh>& GetMesh() const;

    void SetSurfaceColor(const Color& color);

    Color GetSurfaceColor() const;

    void SetMaterial(const Ref<Material>& material);

    const Ref<Material>& GetMaterial() const;

    void SetMeshMaterial(const Ref<Material>& material);

    Ref<Material> GetMeshMaterial() const;

    Ref<Material> GetActiveMaterial() const;

private:
    Ref<Mesh> mesh_{nullptr};
    Ref<Material> material_{nullptr};
    Color surface_color_{0.66f, 0.78f, 0.95f, 1.0f};
};

}
