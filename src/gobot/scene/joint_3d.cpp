/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
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

Affine3 Joint3D::GetJointMotionTransform(RealType joint_position) const {
    Affine3 motion = Affine3::Identity();
    if (joint_type_ == JointType::Revolute || joint_type_ == JointType::Continuous) {
        motion.linear() = AngleAxis(joint_position, axis_).toRotationMatrix();
    } else if (joint_type_ == JointType::Prismatic) {
        motion.translation() = axis_ * joint_position;
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

    const Affine3 reference_motion = GetJointMotionTransform(motion_reference_position_);
    const Affine3 current_motion = GetJointMotionTransform(joint_position_);
    const Affine3 motion_delta = current_motion * reference_motion.inverse();
    for (const auto& [child, child_assembly_transform] : child_assembly_transforms_) {
        if (child) {
            child->SetTransform(motion_delta * child_assembly_transform);
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

void Joint3D::SetInitialPosition(RealType initial_position) {
    initial_position_ = ClampJointPosition(initial_position);
}

RealType Joint3D::GetInitialPosition() const {
    return initial_position_;
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
    motion_reference_position_ = joint_position_;
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

void Joint3D::SetDriveMode(JointDriveMode drive_mode) {
    drive_mode_ = drive_mode;
}

JointDriveMode Joint3D::GetDriveMode() const {
    return drive_mode_;
}

void Joint3D::SetDriveStiffness(RealType stiffness) {
    if (stiffness < 0.0) {
        LOG_ERROR("Joint3D drive stiffness cannot be negative.");
        return;
    }
    drive_stiffness_ = stiffness;
}

RealType Joint3D::GetDriveStiffness() const {
    return drive_stiffness_;
}

void Joint3D::SetDriveDamping(RealType damping) {
    if (damping < 0.0) {
        LOG_ERROR("Joint3D drive damping cannot be negative.");
        return;
    }
    drive_damping_ = damping;
}

RealType Joint3D::GetDriveDamping() const {
    return drive_damping_;
}

void Joint3D::SetControlLowerLimit(RealType lower_limit) {
    control_lower_limit_ = lower_limit;
}

RealType Joint3D::GetControlLowerLimit() const {
    return control_lower_limit_;
}

void Joint3D::SetControlUpperLimit(RealType upper_limit) {
    control_upper_limit_ = upper_limit;
}

RealType Joint3D::GetControlUpperLimit() const {
    return control_upper_limit_;
}

void Joint3D::SetForceLowerLimit(RealType lower_limit) {
    force_lower_limit_ = lower_limit;
}

RealType Joint3D::GetForceLowerLimit() const {
    return force_lower_limit_;
}

void Joint3D::SetForceUpperLimit(RealType upper_limit) {
    force_upper_limit_ = upper_limit;
}

RealType Joint3D::GetForceUpperLimit() const {
    return force_upper_limit_;
}

void Joint3D::SetGear(const std::vector<RealType>& gear) {
    gear_ = gear;
}

const std::vector<RealType>& Joint3D::GetGear() const {
    return gear_;
}

bool Joint3D::HasDrive() const {
    return drive_mode_ != JointDriveMode::Passive;
}

} // namespace gobot

GOBOT_REGISTRATION {

    QuickEnumeration_<JointType>("JointType");
    QuickEnumeration_<JointDriveMode>("JointDriveMode");

    Class_<Joint3D>("Joint3D")
            .constructor()(CtorAsRawPtr)
            .property("joint_type", &Joint3D::GetJointType, &Joint3D::SetJointType)
            .property("parent_link", &Joint3D::GetParentLink, &Joint3D::SetParentLink)
            .property("child_link", &Joint3D::GetChildLink, &Joint3D::SetChildLink)
            .property("axis", &Joint3D::GetAxis, &Joint3D::SetAxis)
            .property("joint_position", &Joint3D::GetJointPosition, &Joint3D::SetJointPosition)
            .property("initial_position", &Joint3D::GetInitialPosition, &Joint3D::SetInitialPosition)
            .property("lower_limit", &Joint3D::GetLowerLimit, &Joint3D::SetLowerLimit)
            .property("upper_limit", &Joint3D::GetUpperLimit, &Joint3D::SetUpperLimit)
            .property("effort_limit", &Joint3D::GetEffortLimit, &Joint3D::SetEffortLimit)
            .property("velocity_limit", &Joint3D::GetVelocityLimit, &Joint3D::SetVelocityLimit)
            .property("damping", &Joint3D::GetDamping, &Joint3D::SetDamping)
            .property("drive_mode", &Joint3D::GetDriveMode, &Joint3D::SetDriveMode)
            .property("drive_stiffness", &Joint3D::GetDriveStiffness, &Joint3D::SetDriveStiffness)
            .property("drive_damping", &Joint3D::GetDriveDamping, &Joint3D::SetDriveDamping)
            .property("control_lower_limit", &Joint3D::GetControlLowerLimit, &Joint3D::SetControlLowerLimit)
            .property("control_upper_limit", &Joint3D::GetControlUpperLimit, &Joint3D::SetControlUpperLimit)
            .property("force_lower_limit", &Joint3D::GetForceLowerLimit, &Joint3D::SetForceLowerLimit)
            .property("force_upper_limit", &Joint3D::GetForceUpperLimit, &Joint3D::SetForceUpperLimit)
            .property("gear", &Joint3D::GetGear, &Joint3D::SetGear);

};
