/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include "gobot/core/color.hpp"
#include "gobot/scene/node_3d.hpp"
#include "gobot/scene/resources/surface_mesh.hpp"
#include "gobot/scene/resources/tetrahedral_mesh.hpp"

namespace gobot {

enum class DeformableBodyModel {
    Volumetric,
    ThinShell,
};

class GOBOT_EXPORT DeformableBody3D : public Node3D {
    GOBCLASS(DeformableBody3D, Node3D)

public:
    DeformableBody3D() = default;

    void SetMesh(const Ref<TetrahedralMesh>& mesh);
    const Ref<TetrahedralMesh>& GetMesh() const;

    void SetSurfaceMesh(const Ref<SurfaceMesh>& mesh);
    const Ref<SurfaceMesh>& GetSurfaceMesh() const;

    void SetModel(DeformableBodyModel model);
    DeformableBodyModel GetModel() const;

    void SetDensity(RealType density);
    RealType GetDensity() const;

    void SetYoungModulus(RealType young_modulus);
    RealType GetYoungModulus() const;

    void SetPoissonRatio(RealType poisson_ratio);
    RealType GetPoissonRatio() const;

    void SetDamping(RealType damping);
    RealType GetDamping() const;

    void SetThickness(RealType thickness);
    RealType GetThickness() const;

    void SetBendingStiffness(RealType stiffness);
    RealType GetBendingStiffness() const;

    void SetKinematic(bool kinematic);
    bool IsKinematic() const;

    void SetCollisionLayer(std::uint32_t collision_layer);
    std::uint32_t GetCollisionLayer() const;

    void SetCollisionMask(std::uint32_t collision_mask);
    std::uint32_t GetCollisionMask() const;

    void SetSelfCollisionEnabled(bool enabled);
    bool IsSelfCollisionEnabled() const;

    void SetDebugSurfaceColor(const Color& color);
    Color GetDebugSurfaceColor() const;

    void SetDebugWireframeVisible(bool visible);
    bool IsDebugWireframeVisible() const;

    void SetRuntimeVertices(const std::vector<Vector3>& vertices);
    const std::vector<Vector3>& GetRuntimeVertices() const;
    void ClearRuntimeVertices();

private:
    Ref<TetrahedralMesh> mesh_;
    Ref<SurfaceMesh> surface_mesh_;
    DeformableBodyModel model_{DeformableBodyModel::Volumetric};
    RealType density_{1000.0};
    RealType young_modulus_{100000.0};
    RealType poisson_ratio_{0.45};
    RealType damping_{0.0};
    RealType thickness_{0.001};
    RealType bending_stiffness_{0.001};
    bool kinematic_{false};
    std::uint32_t collision_layer_{1};
    std::uint32_t collision_mask_{0xffffffffU};
    bool self_collision_enabled_{false};
    Color debug_surface_color_{0.12f, 0.78f, 0.58f, 0.55f};
    bool debug_wireframe_visible_{true};
    std::vector<Vector3> runtime_vertices_;
};

} // namespace gobot
