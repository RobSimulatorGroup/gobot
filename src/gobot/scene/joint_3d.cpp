/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/joint_3d.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

void Joint3D::SetJointType(JointType joint_type) {
    joint_type_ = joint_type;
}

JointType Joint3D::GetJointType() const {
    return joint_type_;
}

void Joint3D::SetParentLink(std::string parent_link) {
    parent_link_ = std::move(parent_link);
}

const std::string& Joint3D::GetParentLink() const {
    return parent_link_;
}

void Joint3D::SetChildLink(std::string child_link) {
    child_link_ = std::move(child_link);
}

const std::string& Joint3D::GetChildLink() const {
    return child_link_;
}

void Joint3D::SetAxis(const Vector3& axis) {
    if (axis.isZero(CMP_EPSILON)) {
        LOG_ERROR("Joint3D axis cannot be zero.");
        return;
    }
    axis_ = axis.normalized();
}

const Vector3& Joint3D::GetAxis() const {
    return axis_;
}

void Joint3D::SetLowerLimit(RealType lower_limit) {
    lower_limit_ = lower_limit;
    SetJointPosition(joint_position_);
}

RealType Joint3D::GetLowerLimit() const {
    return lower_limit_;
}

void Joint3D::SetUpperLimit(RealType upper_limit) {
    upper_limit_ = upper_limit;
    SetJointPosition(joint_position_);
}

RealType Joint3D::GetUpperLimit() const {
    return upper_limit_;
}

void Joint3D::SetEffortLimit(RealType effort_limit) {
    if (effort_limit < 0) {
        LOG_ERROR("Joint3D effort limit cannot be negative.");
        return;
    }
    effort_limit_ = effort_limit;
}

RealType Joint3D::GetEffortLimit() const {
    return effort_limit_;
}

void Joint3D::SetVelocityLimit(RealType velocity_limit) {
    if (velocity_limit < 0) {
        LOG_ERROR("Joint3D velocity limit cannot be negative.");
        return;
    }
    velocity_limit_ = velocity_limit;
}

RealType Joint3D::GetVelocityLimit() const {
    return velocity_limit_;
}

void Joint3D::SetDamping(RealType damping) {
    if (damping < 0) {
        LOG_ERROR("Joint3D damping cannot be negative.");
        return;
    }
    damping_ = damping;
}

RealType Joint3D::GetDamping() const {
    return damping_;
}

RealType Joint3D::ClampJointPosition(RealType joint_position) const {
    if (HasJointPositionLimits()) {
        return std::clamp(joint_position, lower_limit_, upper_limit_);
    }

    return joint_position;
}

Affine3 Joint3D::GetJointMotionTransform() const {
    Affine3 motion = Affine3::Identity();
    if (joint_type_ == JointType::Revolute || joint_type_ == JointType::Continuous) {
        motion.linear() = AngleAxis(joint_position_, axis_).toRotationMatrix();
    } else if (joint_type_ == JointType::Prismatic) {
        motion.translation() = axis_ * joint_position_;
    }

    return motion;
}

void Joint3D::ApplyJointMotion() {
    if (!motion_mode_enabled_) {
        return;
    }

    if (!assembly_pose_valid_) {
        CaptureAssemblyPose();
    }

    SetTransform(assembly_transform_);

    const Affine3 motion = GetJointMotionTransform();
    for (const auto& [child, child_assembly_transform] : child_assembly_transforms_) {
        if (child) {
            child->SetTransform(motion * child_assembly_transform);
        }
    }
}

void Joint3D::SetJointPosition(RealType joint_position) {
    const RealType clamped_position = ClampJointPosition(joint_position);
    joint_position_ = clamped_position;
    ApplyJointMotion();
}

RealType Joint3D::GetJointPosition() const {
    return joint_position_;
}

void Joint3D::ResetJointPosition() {
    joint_position_ = 0.0;
}

bool Joint3D::HasJointPositionLimits() const {
    return (joint_type_ == JointType::Revolute || joint_type_ == JointType::Prismatic) &&
           lower_limit_ < upper_limit_;
}

void Joint3D::CaptureAssemblyPose() {
    assembly_transform_ = GetTransform();
    child_assembly_transforms_.clear();
    child_assembly_transforms_.reserve(GetChildCount());

    for (std::size_t i = 0; i < GetChildCount(); ++i) {
        auto* child_node_3d = Object::PointerCastTo<Node3D>(GetChild(static_cast<int>(i)));
        if (child_node_3d) {
            child_assembly_transforms_.emplace_back(child_node_3d, child_node_3d->GetTransform());
        }
    }

    assembly_pose_valid_ = true;
}

void Joint3D::RestoreAssemblyPose() {
    if (!assembly_pose_valid_) {
        return;
    }

    SetTransform(assembly_transform_);
    for (const auto& [child, child_assembly_transform] : child_assembly_transforms_) {
        if (child) {
            child->SetTransform(child_assembly_transform);
        }
    }
}

void Joint3D::SetMotionModeEnabled(bool enabled) {
    if (motion_mode_enabled_ == enabled) {
        return;
    }

    if (enabled) {
        CaptureAssemblyPose();
        motion_mode_enabled_ = true;
        ApplyJointMotion();
    } else {
        RestoreAssemblyPose();
        motion_mode_enabled_ = false;
    }
}

bool Joint3D::IsMotionModeEnabled() const {
    return motion_mode_enabled_;
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<JointType>("JointType");

    Class_<Joint3D>("Joint3D")
            .constructor()(CtorAsRawPtr)
            .property("joint_type", &Joint3D::GetJointType, &Joint3D::SetJointType)
            .property("parent_link", &Joint3D::GetParentLink, &Joint3D::SetParentLink)
            .property("child_link", &Joint3D::GetChildLink, &Joint3D::SetChildLink)
            .property("axis", &Joint3D::GetAxis, &Joint3D::SetAxis)
            .property("joint_position", &Joint3D::GetJointPosition, &Joint3D::SetJointPosition)
            .property("lower_limit", &Joint3D::GetLowerLimit, &Joint3D::SetLowerLimit)
            .property("upper_limit", &Joint3D::GetUpperLimit, &Joint3D::SetUpperLimit)
            .property("effort_limit", &Joint3D::GetEffortLimit, &Joint3D::SetEffortLimit)
            .property("velocity_limit", &Joint3D::GetVelocityLimit, &Joint3D::SetVelocityLimit)
            .property("damping", &Joint3D::GetDamping, &Joint3D::SetDamping);

};
