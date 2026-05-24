/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "gobot/core/math/geometry.hpp"
#include "gobot/scene/node_3d.hpp"

namespace gobot {

enum class JointType {
    Fixed,
    Revolute,
    Continuous,
    Prismatic,
    Floating,
    Planar
};

enum class JointDriveMode {
    Passive,
    Motor,
    Position,
    Velocity
};

class GOBOT_EXPORT Joint3D : public Node3D {
    GOBCLASS(Joint3D, Node3D)

public:
    Joint3D() = default;

    void SetJointType(JointType joint_type);

    JointType GetJointType() const;

    void SetParentLink(std::string parent_link);

    const std::string& GetParentLink() const;

    void SetChildLink(std::string child_link);

    const std::string& GetChildLink() const;

    void SetAxis(const Vector3& axis);

    const Vector3& GetAxis() const;

    void SetLowerLimit(RealType lower_limit);

    RealType GetLowerLimit() const;

    void SetUpperLimit(RealType upper_limit);

    RealType GetUpperLimit() const;

    void SetEffortLimit(RealType effort_limit);

    RealType GetEffortLimit() const;

    void SetVelocityLimit(RealType velocity_limit);

    RealType GetVelocityLimit() const;

    void SetDamping(RealType damping);

    RealType GetDamping() const;

    void SetJointPosition(RealType joint_position);

    RealType GetJointPosition() const;

    void SetInitialPosition(RealType initial_position);

    RealType GetInitialPosition() const;

    void ResetJointPosition();

    bool HasJointPositionLimits() const;

    void CaptureAssemblyPose();

    void RestoreAssemblyPose();

    void SetMotionModeEnabled(bool enabled);

    bool IsMotionModeEnabled() const;

    void SetDriveMode(JointDriveMode drive_mode);

    JointDriveMode GetDriveMode() const;

    void SetDriveStiffness(RealType stiffness);

    RealType GetDriveStiffness() const;

    void SetDriveDamping(RealType damping);

    RealType GetDriveDamping() const;

    void SetControlLowerLimit(RealType lower_limit);

    RealType GetControlLowerLimit() const;

    void SetControlUpperLimit(RealType upper_limit);

    RealType GetControlUpperLimit() const;

    void SetForceLowerLimit(RealType lower_limit);

    RealType GetForceLowerLimit() const;

    void SetForceUpperLimit(RealType upper_limit);

    RealType GetForceUpperLimit() const;

    void SetGear(const std::vector<RealType>& gear);

    const std::vector<RealType>& GetGear() const;

    bool HasDrive() const;

private:
    RealType ClampJointPosition(RealType joint_position) const;

    Affine3 GetJointMotionTransform(RealType joint_position) const;

    void ApplyJointMotion();

    JointType joint_type_{JointType::Fixed};
    std::string parent_link_;
    std::string child_link_;
    Vector3 axis_{Vector3::UnitX()};
    RealType lower_limit_{0.0};
    RealType upper_limit_{0.0};
    RealType effort_limit_{0.0};
    RealType velocity_limit_{0.0};
    RealType damping_{0.0};
    RealType joint_position_{0.0};
    RealType initial_position_{0.0};
    RealType motion_reference_position_{0.0};
    Affine3 assembly_transform_{Affine3::Identity()};
    std::vector<std::pair<Node3D*, Affine3>> child_assembly_transforms_;
    bool assembly_pose_valid_{false};
    bool motion_mode_enabled_{false};
    JointDriveMode drive_mode_{JointDriveMode::Passive};
    RealType drive_stiffness_{0.0};
    RealType drive_damping_{0.0};
    RealType control_lower_limit_{0.0};
    RealType control_upper_limit_{0.0};
    RealType force_lower_limit_{0.0};
    RealType force_upper_limit_{0.0};
    std::vector<RealType> gear_{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

}
