/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/deformable_attachment_3d.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

void DeformableAttachment3D::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

bool DeformableAttachment3D::IsEnabled() const {
    return enabled_;
}

void DeformableAttachment3D::SetDeformableBodyPath(const NodePath& path) {
    deformable_body_path_ = path;
}

const NodePath& DeformableAttachment3D::GetDeformableBodyPath() const {
    return deformable_body_path_;
}

void DeformableAttachment3D::SetRigidLinkPath(const NodePath& path) {
    rigid_link_path_ = path;
}

const NodePath& DeformableAttachment3D::GetRigidLinkPath() const {
    return rigid_link_path_;
}

void DeformableAttachment3D::SetVertexIndices(
        const std::vector<std::uint32_t>& vertex_indices) {
    vertex_indices_ = vertex_indices;
}

const std::vector<std::uint32_t>& DeformableAttachment3D::GetVertexIndices() const {
    return vertex_indices_;
}

void DeformableAttachment3D::SetStrengthRate(RealType strength_rate) {
    strength_rate_ = strength_rate;
}

RealType DeformableAttachment3D::GetStrengthRate() const {
    return strength_rate_;
}

} // namespace gobot

GOBOT_REGISTRATION {
    Class_<gobot::DeformableAttachment3D>("DeformableAttachment3D")
            .constructor()(CtorAsRawPtr)
            .property("enabled", &gobot::DeformableAttachment3D::IsEnabled,
                      &gobot::DeformableAttachment3D::SetEnabled)
            .property("deformable_body_path",
                      &gobot::DeformableAttachment3D::GetDeformableBodyPath,
                      &gobot::DeformableAttachment3D::SetDeformableBodyPath)
            .property("rigid_link_path",
                      &gobot::DeformableAttachment3D::GetRigidLinkPath,
                      &gobot::DeformableAttachment3D::SetRigidLinkPath)
            .property("vertex_indices",
                      &gobot::DeformableAttachment3D::GetVertexIndices,
                      &gobot::DeformableAttachment3D::SetVertexIndices)
            .property("strength_rate",
                      &gobot::DeformableAttachment3D::GetStrengthRate,
                      &gobot::DeformableAttachment3D::SetStrengthRate);
};
