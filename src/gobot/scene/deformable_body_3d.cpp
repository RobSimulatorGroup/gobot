/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/deformable_body_3d.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void DeformableBody3D::SetMesh(const Ref<TetrahedralMesh>& mesh) {
    mesh_ = mesh;
    runtime_vertices_.clear();
}

const Ref<TetrahedralMesh>& DeformableBody3D::GetMesh() const {
    return mesh_;
}

void DeformableBody3D::SetDensity(RealType density) {
    density_ = density;
}

RealType DeformableBody3D::GetDensity() const {
    return density_;
}

void DeformableBody3D::SetYoungModulus(RealType young_modulus) {
    young_modulus_ = young_modulus;
}

RealType DeformableBody3D::GetYoungModulus() const {
    return young_modulus_;
}

void DeformableBody3D::SetPoissonRatio(RealType poisson_ratio) {
    poisson_ratio_ = poisson_ratio;
}

RealType DeformableBody3D::GetPoissonRatio() const {
    return poisson_ratio_;
}

void DeformableBody3D::SetDamping(RealType damping) {
    damping_ = damping;
}

RealType DeformableBody3D::GetDamping() const {
    return damping_;
}

void DeformableBody3D::SetKinematic(bool kinematic) {
    kinematic_ = kinematic;
}

bool DeformableBody3D::IsKinematic() const {
    return kinematic_;
}

void DeformableBody3D::SetCollisionLayer(std::uint32_t collision_layer) {
    collision_layer_ = collision_layer;
}

std::uint32_t DeformableBody3D::GetCollisionLayer() const {
    return collision_layer_;
}

void DeformableBody3D::SetCollisionMask(std::uint32_t collision_mask) {
    collision_mask_ = collision_mask;
}

std::uint32_t DeformableBody3D::GetCollisionMask() const {
    return collision_mask_;
}

void DeformableBody3D::SetSelfCollisionEnabled(bool enabled) {
    self_collision_enabled_ = enabled;
}

bool DeformableBody3D::IsSelfCollisionEnabled() const {
    return self_collision_enabled_;
}

void DeformableBody3D::SetDebugSurfaceColor(const Color& color) {
    debug_surface_color_ = color;
}

Color DeformableBody3D::GetDebugSurfaceColor() const {
    return debug_surface_color_;
}

void DeformableBody3D::SetDebugWireframeVisible(bool visible) {
    debug_wireframe_visible_ = visible;
}

bool DeformableBody3D::IsDebugWireframeVisible() const {
    return debug_wireframe_visible_;
}

void DeformableBody3D::SetRuntimeVertices(const std::vector<Vector3>& vertices) {
    runtime_vertices_ = vertices;
}

const std::vector<Vector3>& DeformableBody3D::GetRuntimeVertices() const {
    return runtime_vertices_;
}

void DeformableBody3D::ClearRuntimeVertices() {
    runtime_vertices_.clear();
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::DeformableBody3D>("DeformableBody3D")
            .constructor()(CtorAsRawPtr)
            .property("mesh", &gobot::DeformableBody3D::GetMesh,
                      &gobot::DeformableBody3D::SetMesh)
            .property("density", &gobot::DeformableBody3D::GetDensity,
                      &gobot::DeformableBody3D::SetDensity)
            .property("young_modulus", &gobot::DeformableBody3D::GetYoungModulus,
                      &gobot::DeformableBody3D::SetYoungModulus)
            .property("poisson_ratio", &gobot::DeformableBody3D::GetPoissonRatio,
                      &gobot::DeformableBody3D::SetPoissonRatio)
            .property("damping", &gobot::DeformableBody3D::GetDamping,
                      &gobot::DeformableBody3D::SetDamping)
            .property("kinematic", &gobot::DeformableBody3D::IsKinematic,
                      &gobot::DeformableBody3D::SetKinematic)
            .property("collision_layer", &gobot::DeformableBody3D::GetCollisionLayer,
                      &gobot::DeformableBody3D::SetCollisionLayer)
            .property("collision_mask", &gobot::DeformableBody3D::GetCollisionMask,
                      &gobot::DeformableBody3D::SetCollisionMask)
            .property("self_collision_enabled",
                      &gobot::DeformableBody3D::IsSelfCollisionEnabled,
                      &gobot::DeformableBody3D::SetSelfCollisionEnabled)
            .property("debug_surface_color",
                      &gobot::DeformableBody3D::GetDebugSurfaceColor,
                      &gobot::DeformableBody3D::SetDebugSurfaceColor)
            .property("debug_wireframe_visible",
                      &gobot::DeformableBody3D::IsDebugWireframeVisible,
                      &gobot::DeformableBody3D::SetDebugWireframeVisible);
};
