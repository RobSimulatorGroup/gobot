/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
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
}

RealType Joint3D::GetLowerLimit() const {
    return lower_limit_;
}

void Joint3D::SetUpperLimit(RealType upper_limit) {
    upper_limit_ = upper_limit;
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

RealType Joint3D::ClampJointPosition(RealType joint_position) const {
    if (joint_type_ == JointType::Revolute && lower_limit_ < upper_limit_) {
        return std::clamp(joint_position, lower_limit_, upper_limit_);
    }

    return joint_position;
}

void Joint3D::SetJointPosition(RealType joint_position) {
    const RealType clamped_position = ClampJointPosition(joint_position);
    const RealType delta = clamped_position - joint_position_;
    if (std::abs(delta) <= CMP_EPSILON) {
        joint_position_ = clamped_position;
        return;
    }

    if (joint_type_ == JointType::Revolute || joint_type_ == JointType::Continuous) {
        Affine3 transform = GetTransform();
        transform.linear() = transform.linear() * AngleAxis(delta, axis_);
        SetTransform(transform);
    } else if (joint_type_ == JointType::Prismatic) {
        Affine3 transform = GetTransform();
        transform.translation() += axis_ * delta;
        SetTransform(transform);
    }

    joint_position_ = clamped_position;
}

RealType Joint3D::GetJointPosition() const {
    return joint_position_;
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
            .property("velocity_limit", &Joint3D::GetVelocityLimit, &Joint3D::SetVelocityLimit);

};
